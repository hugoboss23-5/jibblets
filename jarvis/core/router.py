"""
The Self-Modifying Neural Router — the core of JARVIS.

This is the piece that doesn't exist anywhere else. A neural network that
modifies its own weights during inference. Training and inference are fused
into one loop. The forward pass includes the weight update.

Architecture:
    Input: user query embedding + context vector + conversation history
    -> Chestahedron Embedding (project to 7D manifold)
    -> Fast Weight Layers (Hebbian self-modification)
    -> Hypernetwork Routing (context-conditioned weight generation)
    -> Internal Loss (routing + coherence + quality)
    -> Weight Update (applied to modifiable subset during forward pass)
    -> Output: routing decision + shaped context + meta-parameters

The router operates on the 7-face Chestahedron topology. All 7 cognitive
operations fire simultaneously, weighted by need. The geometric coherence
score constrains self-modification to stay on-manifold.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chestahedron import ChestahedronEmbedding, ChestahedronProjector, NUM_FACES, FACE_NAMES
from .fast_weights import FastWeightLayer
from .hypernetwork import RoutingHyperNetwork
from .loss import InternalLoss
from .safety import WeightSafetyGuard

logger = logging.getLogger(__name__)


class RoutingDecision:
    """Represents the output of a routing decision."""
    ROUTES = ["LOCAL", "MEMORY", "API_SONNET", "API_OPUS", "MEMORY_THEN_API"]

    def __init__(
        self,
        route: str,
        route_probs: torch.Tensor,
        face_activations: torch.Tensor,
        shaped_context: dict,
        meta: dict,
    ):
        self.route = route
        self.route_probs = route_probs
        self.face_activations = face_activations
        self.shaped_context = shaped_context
        self.meta = meta

    def __repr__(self):
        top_face_idx = self.face_activations.argmax().item()
        return (
            f"RoutingDecision(route={self.route}, "
            f"dominant_face={FACE_NAMES[top_face_idx]}, "
            f"confidence={self.route_probs.max().item():.3f})"
        )


@dataclass
class RouterConfig:
    """Configuration for the self-modifying router."""
    input_dim: int = 384          # Input embedding dimension
    hidden_dim: int = 256         # Internal hidden dimension
    num_routes: int = 5           # Number of routing options
    num_fast_layers: int = 2      # Number of fast-weight layers
    num_heads: int = 4            # Attention heads in fast weights
    hyper_hidden: int = 128       # Hypernetwork hidden dimension
    num_hyper_layers: int = 3     # Hypernetwork routing layers
    fast_weight_decay: float = 0.95
    fast_weight_lr: float = 0.5
    internal_lr: float = 0.001    # Learning rate for internal optimizer
    coherence_weight: float = 1.0
    adjacency_weight: float = 0.5
    routing_weight: float = 1.0
    quality_weight: float = 1.0
    checkpoint_every: int = 50    # Checkpoint every N interactions
    max_delta_per_step: float = 1.0
    dropout: float = 0.1


class SelfModifyingRouter(nn.Module):
    """
    The self-modifying neural router.

    The forward pass fuses inference and self-modification:
    1. Encode input into Chestahedron 7D space
    2. Pass through fast-weight layers (Hebbian updates happen here)
    3. Context-condition the routing via hypernetwork
    4. Compute internal loss (routing + coherence + quality)
    5. Apply gradient-based updates to modifiable parameters
    6. Output routing decision + shaped context
    """

    def __init__(self, config: RouterConfig | None = None):
        super().__init__()
        self.config = config or RouterConfig()
        c = self.config

        # --- Input encoding ---
        self.input_encoder = nn.Sequential(
            nn.Linear(c.input_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )

        # --- Chestahedron projection ---
        self.chesta_embed = ChestahedronEmbedding(c.hidden_dim)
        self.chesta_proj = ChestahedronProjector()

        # --- Fast-weight layers (self-modifying) ---
        self.fast_layers = nn.ModuleList([
            FastWeightLayer(c.hidden_dim, num_heads=c.num_heads, dropout=c.dropout)
            for _ in range(c.num_fast_layers)
        ])

        # --- Hypernetwork routing ---
        # Context = 7D Chestahedron state + hidden state
        context_dim = NUM_FACES + c.hidden_dim
        self.routing_hyper = RoutingHyperNetwork(
            context_dim=context_dim,
            routing_dim=c.hidden_dim,
            num_routing_layers=c.num_hyper_layers,
            hyper_hidden=c.hyper_hidden,
        )

        # --- Route output head ---
        self.route_head = nn.Sequential(
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.num_routes),
        )

        # --- Context shaping head (for organ calls) ---
        self.context_shaper = nn.Sequential(
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )

        # --- Internal loss ---
        self.internal_loss = InternalLoss(
            num_routes=c.num_routes,
            routing_weight=c.routing_weight,
            coherence_weight=c.coherence_weight,
            adjacency_weight=c.adjacency_weight,
            quality_weight=c.quality_weight,
        )

        # --- Internal optimizer (differentiable component) ---
        # Only optimizes modifiable parameters
        self._internal_optimizer: torch.optim.Optimizer | None = None

        # --- State tracking ---
        self._step_count = 0
        self._last_route_idx: int | None = None
        self._safety_guard: WeightSafetyGuard | None = None
        self._modification_log: list[dict] = []

    def _get_internal_optimizer(self) -> torch.optim.Optimizer:
        """Lazy-init the internal optimizer over modifiable params only."""
        if self._internal_optimizer is None:
            modifiable_params = []
            for name, param in self.named_parameters():
                if param.requires_grad and self._is_modifiable(name):
                    modifiable_params.append(param)
            self._internal_optimizer = torch.optim.Adam(
                modifiable_params, lr=self.config.internal_lr
            )
        return self._internal_optimizer

    def _is_modifiable(self, name: str) -> bool:
        """Check if a parameter is modifiable (not frozen safety weights)."""
        if self._safety_guard:
            return self._safety_guard.is_modifiable(name)
        # Default: fast weight and hyper components are modifiable
        modifiable_keywords = [
            "fast_", "hyper", "routing_hyper", "route_head",
            "context_shaper", "chesta_embed", "gate", "scale",
            "lr_gate", "decay_gate",
        ]
        return any(kw in name for kw in modifiable_keywords)

    def init_safety(self, checkpoint_dir: str | None = None):
        """Initialize the safety guard for weight modification boundaries."""
        self._safety_guard = WeightSafetyGuard(
            model=self,
            modifiable_prefixes=[
                "fast_layers", "routing_hyper", "route_head",
                "context_shaper", "chesta_embed",
            ],
            frozen_prefixes=[
                "input_encoder",  # Input encoding is stable
            ],
            max_delta_per_step=self.config.max_delta_per_step,
            checkpoint_dir=checkpoint_dir,
        )

    def forward(
        self,
        x: torch.Tensor,
        quality_signal: float | None = None,
        self_modify: bool = True,
    ) -> RoutingDecision:
        """
        Forward pass with fused inference and self-modification.

        Args:
            x: (batch, input_dim) input embedding (user query + context)
            quality_signal: optional quality feedback from previous interaction [0, 1]
            self_modify: whether to perform self-modification this pass

        Returns:
            RoutingDecision with route, probabilities, face activations, and shaped context
        """
        self._step_count += 1
        batch_size = x.shape[0]
        mod_info = {}

        # 1. Encode input
        hidden = self.input_encoder(x)  # (B, hidden_dim)

        # 2. Project to Chestahedron 7D space
        state_7d = self.chesta_embed(hidden)  # (B, 7)

        # 3. Pass through fast-weight layers (read only — no Hebbian update yet)
        # Hebbian updates are deferred until after backward pass to avoid
        # in-place modification conflicts with autograd.
        fast_info_list = []
        for layer in self.fast_layers:
            hidden, fast_info = layer(hidden, update_fast=False)
            fast_info_list.append(fast_info)

        # 4. Context-conditioned routing via hypernetwork
        # Context = Chestahedron state concatenated with hidden state
        context = torch.cat([state_7d, hidden], dim=-1)  # (B, 7 + hidden_dim)
        routed = self.routing_hyper(hidden, context)  # (B, hidden_dim)

        # 5. Compute route logits and shaped context
        route_logits = self.route_head(routed)  # (B, num_routes)
        shaped_context_vec = self.context_shaper(routed)  # (B, hidden_dim)

        # 6. Compute internal loss and self-modify
        if self_modify:
            loss, loss_info = self.internal_loss(
                route_logits=route_logits,
                state_7d=state_7d,
                quality_signal=quality_signal,
                selected_route=self._last_route_idx,
            )

            # Apply internal optimizer step (gradient-based self-modification)
            optimizer = self._get_internal_optimizer()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            # Now apply Hebbian fast-weight updates (non-differentiable, after backward)
            hidden_det = hidden.detach()
            for layer in self.fast_layers:
                layer.fast_memory(hidden_det, update=True)

            fast_info_list = [
                {"updated": True, "fast_weight_norm": l.fast_memory.get_fast_weight_norm()}
                for l in self.fast_layers
            ]

            mod_info = {
                "loss": loss_info,
                "fast_weights": fast_info_list,
                "step": self._step_count,
            }

            # Checkpoint if needed
            if (
                self._safety_guard
                and self._step_count % self.config.checkpoint_every == 0
            ):
                self._safety_guard.checkpoint()

        # 7. Select route (argmax for execution, but keep probs for the decision)
        route_probs = F.softmax(route_logits.detach(), dim=-1)
        route_idx = route_probs.argmax(dim=-1).item()
        self._last_route_idx = route_idx

        # 8. Get face activations for the decision
        face_acts = self.chesta_proj.face_activations(state_7d.detach())

        # 9. Build shaped context dict for the organ
        operation_weights = face_acts.squeeze(0).tolist() if face_acts.dim() > 1 else face_acts.tolist()
        shaped_context = {
            "context_vector": shaped_context_vec.detach(),
            "operation_weights": {
                FACE_NAMES[i]: w for i, w in enumerate(operation_weights)
            },
            "dominant_operation": FACE_NAMES[face_acts.argmax(dim=-1).item()],
            "coherence": self.chesta_proj.coherence_score(state_7d).mean().item(),
            "route_confidence": route_probs.max().item(),
        }

        decision = RoutingDecision(
            route=RoutingDecision.ROUTES[route_idx],
            route_probs=route_probs.squeeze(0),
            face_activations=face_acts.squeeze(0),
            shaped_context=shaped_context,
            meta=mod_info,
        )

        if mod_info:
            self._modification_log.append({
                "step": self._step_count,
                "route": decision.route,
                "coherence": shaped_context["coherence"],
                "loss": mod_info.get("loss", {}).get("total_loss", 0),
            })

        return decision

    def get_modification_history(self) -> list[dict]:
        """Return the log of all self-modifications."""
        return self._modification_log.copy()

    def save_state(self, path: str | Path):
        """Save full router state (weights + fast weights + optimizer)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "optimizer_state": (
                self._internal_optimizer.state_dict()
                if self._internal_optimizer else None
            ),
            "step_count": self._step_count,
            "config": self.config,
            "modification_log": self._modification_log,
        }, path / "router_state.pt")
        logger.info(f"Router state saved to {path}")

    def load_state(self, path: str | Path):
        """Load router state from disk."""
        path = Path(path)
        state_file = path / "router_state.pt"
        if not state_file.exists():
            logger.warning(f"No state file at {state_file}")
            return

        checkpoint = torch.load(state_file, weights_only=False)
        self.load_state_dict(checkpoint["model_state"])
        self._step_count = checkpoint.get("step_count", 0)
        self._modification_log = checkpoint.get("modification_log", [])

        if checkpoint.get("optimizer_state") and self._internal_optimizer:
            self._internal_optimizer.load_state_dict(checkpoint["optimizer_state"])

        logger.info(f"Router state loaded from {path} (step {self._step_count})")


class SimpleTextEncoder(nn.Module):
    """
    Simple text encoder for converting raw text to input embeddings.
    Uses a small learned vocabulary embedding + positional encoding.
    This is intentionally simple — the intelligence comes from the organs.
    """

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 384, max_len: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.pool_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token IDs to a single embedding vector.

        Args:
            token_ids: (batch, seq_len) integer token IDs

        Returns:
            (batch, embed_dim) pooled embedding
        """
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)

        x = self.token_embed(token_ids) + self.pos_embed(positions)
        # Mean pool over sequence
        pooled = x.mean(dim=1)
        return self.norm(self.pool_proj(pooled))
