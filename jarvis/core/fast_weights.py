"""
Fast Weights module — Hebbian/outer-product weight updates during forward pass.

This implements the key self-modification mechanism: a fast-weight matrix that
is updated every forward pass via outer product of activations. No backprop
needed for the update itself. The fast weights augment (not replace) the slow
learned weights, allowing real-time adaptation without offline training.

Reference: Ba et al., "Using Fast Weights to Attend to the Recent Past" (2016)
Extended with decay, gating, and capacity constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastWeightMemory(nn.Module):
    """
    Fast associative memory that self-modifies during inference.

    Maintains a fast-weight matrix A that is updated each forward pass:
        A_t = decay * A_{t-1} + lr * outer(key, value)

    The fast weights store recent activation patterns and allow the router
    to adapt its behavior in real-time based on recent interactions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        decay: float = 0.95,
        fast_lr: float = 0.5,
        max_norm: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.decay = decay
        self.fast_lr = fast_lr
        self.max_norm = max_norm

        # Projections for key/value/query
        self.to_keys = nn.Linear(dim, dim, bias=False)
        self.to_values = nn.Linear(dim, dim, bias=False)
        self.to_queries = nn.Linear(dim, dim, bias=False)

        # Learnable decay and learning rate (per-head)
        self.decay_gate = nn.Parameter(torch.full((num_heads,), decay))
        self.lr_gate = nn.Parameter(torch.full((num_heads,), fast_lr))

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(dim)

        # The fast weight matrix — NOT a parameter, it's runtime state
        # Shape: (num_heads, head_dim, head_dim)
        self.register_buffer(
            "fast_weights",
            torch.zeros(num_heads, self.head_dim, self.head_dim),
        )

        # Modification log for safety/reversibility
        self._update_history: list[torch.Tensor] = []
        self._max_history = 100

    def reset_fast_weights(self):
        """Reset fast weights to zero (fresh start)."""
        self.fast_weights.zero_()
        self._update_history.clear()

    def get_fast_weight_norm(self) -> float:
        """Get the current norm of fast weights for monitoring."""
        return self.fast_weights.norm().item()

    def forward(
        self, x: torch.Tensor, update: bool = True
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass with optional fast-weight self-modification.

        Args:
            x: (batch, dim) input activations
            update: whether to update fast weights this pass

        Returns:
            output: (batch, dim) transformed output
            info: dict with modification metadata
        """
        batch_size = x.shape[0]

        # Project to keys, values, queries
        k = self.to_keys(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.to_values(x).view(batch_size, self.num_heads, self.head_dim)
        q = self.to_queries(x).view(batch_size, self.num_heads, self.head_dim)

        # Normalize keys for stable associative storage
        k = F.normalize(k, dim=-1)

        # --- READ from fast weights ---
        # query the fast weight memory: output = A @ q
        # fast_weights: (H, D, D), q: (B, H, D) -> (B, H, D)
        read_out = torch.einsum("hde,bhe->bhd", self.fast_weights, q)

        # Combine read-out with values via gating
        gate = torch.sigmoid(read_out)
        combined = gate * read_out + (1 - gate) * v

        # --- WRITE to fast weights (the self-modification) ---
        # Hebbian update is non-differentiable by design — it operates
        # outside the autograd graph. Gradient-based learning happens
        # through the internal optimizer on the slow/modifiable weights.
        info = {"updated": False, "fast_weight_norm": 0.0, "update_norm": 0.0}

        if update:
            with torch.no_grad():
                # Compute Hebbian update: outer product of keys and values
                k_det = k.detach()
                v_det = v.detach()
                update_matrix = torch.einsum("bhk,bhv->hkv", k_det, v_det) / batch_size

                # Apply learnable decay and learning rate
                effective_decay = torch.sigmoid(self.decay_gate).view(-1, 1, 1)
                effective_lr = torch.sigmoid(self.lr_gate).view(-1, 1, 1)

                # Store pre-update state for reversibility
                if len(self._update_history) < self._max_history:
                    self._update_history.append(self.fast_weights.clone())
                else:
                    self._update_history.pop(0)
                    self._update_history.append(self.fast_weights.clone())

                # Update: A_t = decay * A_{t-1} + lr * outer(k, v)
                new_fw = effective_decay * self.fast_weights + effective_lr * update_matrix

                # Norm constraint: prevent fast weights from exploding
                fw_norm = new_fw.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
                scale = torch.clamp(self.max_norm / fw_norm, max=1.0)
                new_fw = new_fw * scale

                self.fast_weights.copy_(new_fw)

                info["updated"] = True
                info["fast_weight_norm"] = new_fw.norm().item()
                info["update_norm"] = update_matrix.norm().item()

        # Reshape and project output
        combined = combined.reshape(batch_size, self.dim)
        output = self.out_proj(self.norm(combined))

        return output, info

    def revert_last_update(self) -> bool:
        """Revert to the previous fast weight state. Returns True if successful."""
        if self._update_history:
            self.fast_weights.copy_(self._update_history.pop())
            return True
        return False


class FastWeightLayer(nn.Module):
    """
    A complete fast-weight augmented layer: slow weights + fast weights + residual.
    This is the building block for the self-modifying router.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fast_memory = FastWeightMemory(dim, num_heads=num_heads)
        self.slow_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, update_fast: bool = True
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward with residual connections around both fast and slow paths.
        """
        # Fast-weight path (self-modifying)
        normed = self.norm1(x)
        fast_out, info = self.fast_memory(normed, update=update_fast)
        x = x + self.dropout(fast_out)

        # Slow-weight path (standard FFN)
        x = x + self.dropout(self.slow_ffn(self.norm2(x)))

        return x, info
