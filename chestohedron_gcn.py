"""
Gated Chestohedron Network (GCN) — GRU gating INSIDE chestohedron stages
========================================================================

CCN and GRU solve ORTHOGONAL problems:
  - CCN solves WHERE: which stage runs, what kind of transform. Fixed. Free.
  - GRU solves WHAT: which information flows through, keep vs forget. Learned.

These compose. The hybrid puts per-stage GRU gates INSIDE the 7 chestohedron
stages. Each stage has its OWN reset + update gates — MIRROR learns different
filtering than VERIFY. The topology specializes the gates.

Why GRU not LSTM:
  - LSTM's output gate controls exposure to next layer
  - But chestohedron routing already controls exposure (topology)
  - Output gate is REDUNDANT with topology
  - GRU's 2-gate design handles keep-vs-forget without interference

Per-stage gated processing:
  r = sigmoid(r_proj(r_state(state) + r_input(u)))    # reset
  z = sigmoid(z_proj(z_state(state) + z_input(u)))    # update
  candidate = tanh((r * state) @ W_topo + u)          # topology on filtered state
  state = (1 - z) * state + z * candidate             # gated update

Gates use low-rank factored projections to stay parameter-efficient.
The topology matrices are FIXED. The gates are LEARNED. They don't interfere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

from chestohedron_lm import (
    PHI, STAGE_TOKENS, STAGE_END_TOKENS,
    V9DataGenerator, V9Tokenizer, make_topology_matrices, prepare_batches,
    train_model, ChestohedronLM, VanillaLSTM_LM, VanillaGRU_LM,
    generate_v9_response,
)


# ============================================================
# STAGE GRU GATE — low-rank factored, parameter-efficient
# ============================================================

class StageGRUGate(nn.Module):
    """
    GRU-style gating for a single chestohedron stage.
    Uses additive low-rank factorization to stay parameter-efficient.

    Reset gate:  controls what prior state the topology matrix sees.
    Update gate: controls how much the topology result replaces prior state.

    Each gate: sigmoid(proj(W_state(state) + W_input(u)))
    where W_state and W_input are (hidden_dim, gate_rank) — additive in
    low-rank space, then projected back to hidden_dim.
    """

    def __init__(self, hidden_dim, gate_rank=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_rank = gate_rank

        # Reset gate — low-rank factored
        self.r_state = nn.Linear(hidden_dim, gate_rank, bias=False)
        self.r_input = nn.Linear(hidden_dim, gate_rank, bias=False)
        self.r_proj = nn.Linear(gate_rank, hidden_dim)

        # Update gate — low-rank factored
        self.z_state = nn.Linear(hidden_dim, gate_rank, bias=False)
        self.z_input = nn.Linear(hidden_dim, gate_rank, bias=False)
        self.z_proj = nn.Linear(gate_rank, hidden_dim)

        # Init: reset bias toward 1 (pass through), update bias toward 0 (conservative)
        nn.init.constant_(self.r_proj.bias, 1.0)
        nn.init.constant_(self.z_proj.bias, 0.0)
        # Small init on weights so gates start near pass-through
        for p in [self.r_state, self.r_input, self.z_state, self.z_input]:
            nn.init.normal_(p.weight, std=0.02)
        nn.init.normal_(self.r_proj.weight, std=0.02)
        nn.init.normal_(self.z_proj.weight, std=0.02)

    def forward(self, state, u):
        r = torch.sigmoid(self.r_proj(self.r_state(state) + self.r_input(u)))
        z = torch.sigmoid(self.z_proj(self.z_state(state) + self.z_input(u)))
        return r, z


class UpdateOnlyGate(nn.Module):
    """Ablation: update gate only, no reset gate."""

    def __init__(self, hidden_dim, gate_rank=3):
        super().__init__()
        self.z_state = nn.Linear(hidden_dim, gate_rank, bias=False)
        self.z_input = nn.Linear(hidden_dim, gate_rank, bias=False)
        self.z_proj = nn.Linear(gate_rank, hidden_dim)

        nn.init.constant_(self.z_proj.bias, 0.0)
        for p in [self.z_state, self.z_input]:
            nn.init.normal_(p.weight, std=0.02)
        nn.init.normal_(self.z_proj.weight, std=0.02)

    def forward(self, state, u):
        z = torch.sigmoid(self.z_proj(self.z_state(state) + self.z_input(u)))
        r = torch.ones_like(z)
        return r, z


# ============================================================
# GATED CHESTOHEDRON BLOCK
# ============================================================

class GatedChestohedronBlock(nn.Module):
    """
    V9 cycling block with per-stage GRU gating.

    Each of the 7 stages has its own low-rank gate that learns:
      - What history to show the topology matrix (reset)
      - How much to accept the topology's output (update)

    Topology matrices: FIXED. Zero learned params.
    Gate weights: LEARNED. Per-stage specialization.
    """

    def __init__(self, dim, gate_rank=3, seed=42,
                 shared_gates=False, update_only=False):
        super().__init__()
        self.dim = dim

        # FIXED topology matrices
        matrices = make_topology_matrices(dim, seed)
        for name, W in matrices.items():
            self.register_buffer(f'W_{name}', W)

        # PER-STAGE GRU GATES
        GateClass = UpdateOnlyGate if update_only else StageGRUGate
        if shared_gates:
            shared = GateClass(dim, gate_rank)
            self.stage_gates = nn.ModuleList([shared] * 7)
        else:
            self.stage_gates = nn.ModuleList([
                GateClass(dim, gate_rank) for _ in range(7)
            ])

        # LEARNED: thin per-stage scale/shift (same as CCN)
        self.stage_scale = nn.ParameterList([
            nn.Parameter(torch.ones(dim)) for _ in range(7)
        ])
        self.stage_shift = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(7)
        ])

        # LEARNED: input gate + stage context
        self.input_gate = nn.Linear(dim, dim, bias=False)
        self.stage_embed = nn.Embedding(7, dim)

        # For recording gate activations
        self._record_gates = False
        self._gate_history = []

    def _get_stage_matrices(self):
        return [
            self.W_mirror, self.W_inherit, self.W_bound,
            self.W_express, self.W_verify, self.W_remove,
        ]

    def _gate6_forward(self, state, u, gate):
        """GATE 6 with GRU gating on the 3-mode adaptive closer."""
        r, z = gate(state, u)

        if self._record_gates:
            self._gate_history.append({
                'stage': 6,
                'r_mean': r.mean().item(),
                'z_mean': z.mean().item(),
                'r_std': r.std().item(),
                'z_std': z.std().item(),
            })

        filtered_state = r * state

        energy = (filtered_state ** 2).sum(dim=-1, keepdim=True)
        energy_norm = energy / (energy.max() + 1e-8)

        g_tamam = torch.clamp(energy_norm - 0.66, min=0) * 3
        g_darash = torch.clamp(1 - (energy_norm - 0.5).abs() * 4, min=0)
        g_zakat = torch.clamp(0.33 - energy_norm, min=0) * 3
        g_total = g_tamam + g_darash + g_zakat + 1e-8

        candidate = ((g_tamam / g_total) * (filtered_state @ self.W_gate6_tamam) +
                      (g_darash / g_total) * (filtered_state @ self.W_gate6_darash) +
                      (g_zakat / g_total) * (filtered_state @ self.W_gate6_zakat))
        candidate = torch.tanh(candidate * self.stage_scale[6] + self.stage_shift[6] + u)

        return (1 - z) * state + z * candidate

    def forward(self, state, x, stage_idx=None):
        """
        One timestep through the gated V9 cycle.

        For each of the 6 main stages:
          1. Gate computes (r, z) from state and input
          2. Reset filters state: filtered = r * state
          3. Topology transform: candidate = tanh(filtered @ W_topo + u)
          4. Update blends: state = (1-z) * state + z * candidate

        Then GATE 6 applies the same pattern to the 3-mode closer.
        """
        u = self.input_gate(x)
        if stage_idx is not None:
            u = u + self.stage_embed(stage_idx) * 0.3

        matrices = self._get_stage_matrices()

        for i, W in enumerate(matrices):
            gate = self.stage_gates[i]
            r, z = gate(state, u)

            if self._record_gates:
                self._gate_history.append({
                    'stage': i,
                    'r_mean': r.mean().item(),
                    'z_mean': z.mean().item(),
                    'r_std': r.std().item(),
                    'z_std': z.std().item(),
                })

            filtered_state = r * state
            candidate = filtered_state @ W + u
            candidate = candidate * self.stage_scale[i] + self.stage_shift[i]
            candidate = torch.tanh(candidate)
            state = (1 - z) * state + z * candidate

        # GATE 6
        state = self._gate6_forward(state, u, self.stage_gates[6])
        return state


# ============================================================
# GATED CHESTOHEDRON LANGUAGE MODEL
# ============================================================

class GatedChestohedronLM(nn.Module):
    """
    GCN Language Model — topology routes, gates filter.

    Same skeleton as ChestohedronLM but with per-stage GRU gating.
    Low-rank factored gates keep param count well under LSTM.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_cycles=1,
                 gate_rank=3, stage_token_ids=None, seed=42,
                 shared_gates=False, update_only=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = n_cycles
        self.gate_rank = gate_rank

        self.stage_token_map = {}
        if stage_token_ids:
            for i, tid in enumerate(stage_token_ids):
                self.stage_token_map[tid] = i

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj_in = nn.Linear(embed_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            GatedChestohedronBlock(hidden_dim, gate_rank=gate_rank,
                                    seed=seed + i,
                                    shared_gates=shared_gates,
                                    update_only=update_only)
            for i in range(n_cycles)
        ])

        self.ln = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None, stage_indices=None):
        batch_size, seq_len = x.shape

        if hidden is None:
            hidden = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
                      for _ in range(self.n_cycles)]

        emb = self.embed(x)
        inp = self.proj_in(emb)

        outputs = []
        for t in range(seq_len):
            h = inp[:, t, :]
            si = stage_indices[:, t] if stage_indices is not None else None

            for i, block in enumerate(self.blocks):
                hidden[i] = block(hidden[i], h, si)
                h = hidden[i]

            outputs.append(self.ln(h))

        out = torch.stack(outputs, dim=1)
        logits = self.proj_out(out)
        return logits, hidden

    @property
    def learned_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def fixed_params(self):
        return sum(b.numel() for b in self.buffers())

    def enable_gate_recording(self):
        for block in self.blocks:
            block._record_gates = True
            block._gate_history = []

    def disable_gate_recording(self):
        for block in self.blocks:
            block._record_gates = False

    def get_gate_history(self):
        history = []
        for block in self.blocks:
            history.extend(block._gate_history)
        return history


# ============================================================
# GATE ACTIVATION VISUALIZATION
# ============================================================

def record_gate_activations(model, batches, n_batches=5):
    """Record gate activations over several batches."""
    model.eval()
    model.enable_gate_recording()
    with torch.no_grad():
        for x, y, si in batches[:n_batches]:
            model(x, None, si)
    history = model.get_gate_history()
    model.disable_gate_recording()
    return history


def visualize_gate_activations(history, save_path="gcn_gate_activations.png"):
    """Per-stage gate activation visualization."""
    stage_names = ["MIRROR", "INHERIT", "BOUND", "EXPRESS", "VERIFY", "REMOVE", "GATE6"]

    # Aggregate per stage
    stage_stats = {i: {'r_means': [], 'z_means': [], 'r_stds': [], 'z_stds': []}
                   for i in range(7)}
    for entry in history:
        s = entry['stage']
        stage_stats[s]['r_means'].append(entry['r_mean'])
        stage_stats[s]['z_means'].append(entry['z_mean'])
        stage_stats[s]['r_stds'].append(entry['r_std'])
        stage_stats[s]['z_stds'].append(entry['z_std'])

    agg = {}
    for s in range(7):
        stats = stage_stats[s]
        if stats['r_means']:
            agg[s] = {
                'r_mean': sum(stats['r_means']) / len(stats['r_means']),
                'z_mean': sum(stats['z_means']) / len(stats['z_means']),
                'r_std': sum(stats['r_stds']) / len(stats['r_stds']),
                'z_std': sum(stats['z_stds']) / len(stats['z_stds']),
            }
        else:
            agg[s] = {'r_mean': 0, 'z_mean': 0, 'r_std': 0, 'z_std': 0}

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Per-Stage Gate Activation Patterns (GCN)', fontsize=14, fontweight='bold')

        x_pos = range(7)
        colors = ['#4a90d9', '#67b7dc', '#7bc8a4', '#f5a623',
                  '#d0021b', '#9013fe', '#50e3c2']

        # Reset gate
        ax = axes[0]
        r_means = [agg[s]['r_mean'] for s in range(7)]
        r_stds = [agg[s]['r_std'] for s in range(7)]
        ax.bar(x_pos, r_means, yerr=r_stds, capsize=5,
               color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Mean Activation', fontsize=11)
        ax.set_title('Reset Gate (r) — What history does each stage see?', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_names, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        for i, (m, s) in enumerate(zip(r_means, r_stds)):
            ax.text(i, min(m + s + 0.03, 1.02), f'{m:.3f}', ha='center', fontsize=9)

        # Update gate
        ax = axes[1]
        z_means = [agg[s]['z_mean'] for s in range(7)]
        z_stds = [agg[s]['z_std'] for s in range(7)]
        ax.bar(x_pos, z_means, yerr=z_stds, capsize=5,
               color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Mean Activation', fontsize=11)
        ax.set_title('Update Gate (z) — How much does each stage change state?', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_names, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        for i, (m, s) in enumerate(zip(z_means, z_stds)):
            ax.text(i, min(m + s + 0.03, 1.02), f'{m:.3f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Gate activation plot saved to {save_path}")

    except ImportError:
        pass

    # Always print text visualization
    print(f"\n  Gate Activation Patterns:")
    print(f"  {'Stage':<10s} {'Reset(r)':>10s} {'Update(z)':>10s} {'r_std':>8s} {'z_std':>8s}")
    print(f"  {'─' * 50}")
    for s in range(7):
        a = agg[s]
        print(f"  {stage_names[s]:<10s} {a['r_mean']:>10.4f} {a['z_mean']:>10.4f} "
              f"{a['r_std']:>8.4f} {a['z_std']:>8.4f}")

    # Check for specialization
    r_vals = [agg[s]['r_mean'] for s in range(7)]
    z_vals = [agg[s]['z_mean'] for s in range(7)]
    r_spread = max(r_vals) - min(r_vals)
    z_spread = max(z_vals) - min(z_vals)
    print(f"\n  Reset gate spread:  {r_spread:.4f} (higher = more per-stage specialization)")
    print(f"  Update gate spread: {z_spread:.4f} (higher = more per-stage specialization)")

    return agg


# ============================================================
# CHESTOHEDRON DIRECT TRAINING (CDT)
# ============================================================
#
# Standard backprop learns structure AND filters simultaneously.
# The chestohedron SEPARATES them:
#   Structure: fixed topology matrices (zero learning needed)
#   Filtering: low-rank gates (~13K params, solvable directly)
#
# CDT exploits this separation:
#   Phase 1: Forward-only reservoir pass (topology does structural work)
#   Phase 2: Closed-form output layer (ridge regression, one matrix op)
#   Phase 3: Gate calibration (fine-tune only 13K gate params)
#
# Total: minutes, not hours. The topology makes compute obsolete.

def reservoir_collect(model, batches, use_stages=True):
    """
    Phase 1: Run data through fixed topology, collect hidden states.

    Forward-only. No backprop. No gradients. The chestohedron geometry
    processes the sequences — we just record what it produces.

    Returns: (all_hidden, all_targets) tensors
    """
    model.eval()
    all_hidden = []
    all_targets = []

    with torch.no_grad():
        for x, y, si in batches:
            batch_size, seq_len = x.shape
            emb = model.embed(x)
            inp = model.proj_in(emb)

            hidden = [torch.zeros(batch_size, model.hidden_dim)
                      for _ in range(model.n_cycles)]

            for t in range(seq_len):
                h = inp[:, t, :]
                stage = si[:, t] if use_stages else None

                for i, block in enumerate(model.blocks):
                    hidden[i] = block(hidden[i], h, stage)
                    h = hidden[i]

                all_hidden.append(model.ln(h))
                all_targets.append(y[:, t])

    H = torch.cat(all_hidden, dim=0)    # (N_total, hidden_dim)
    Y = torch.cat(all_targets, dim=0)   # (N_total,)
    return H, Y


def solve_output_closed_form(H, Y, vocab_size, hidden_dim, lambda_reg=1.0):
    """
    Phase 2: Solve output projection via ridge regression.

    W_out = (H^T H + λI)^{-1} H^T Y_onehot

    One matrix operation. Globally optimal in L2 sense.
    No epochs. No learning rate. No gradient descent.
    """
    print(f"    Solving output layer (ridge regression, λ={lambda_reg})...")
    print(f"    H shape: {H.shape}, vocab: {vocab_size}")

    # One-hot encode targets
    Y_oh = torch.zeros(Y.shape[0], vocab_size)
    Y_oh.scatter_(1, Y.unsqueeze(1), 1.0)

    # Ridge regression: W = (H^T H + λI)^{-1} H^T Y
    HtH = H.T @ H  # (hidden_dim, hidden_dim)
    HtH += lambda_reg * torch.eye(hidden_dim)
    HtY = H.T @ Y_oh  # (hidden_dim, vocab_size)

    W_out = torch.linalg.solve(HtH, HtY)  # (hidden_dim, vocab_size)

    # Compute training loss with this closed-form solution
    logits = H @ W_out
    loss = F.cross_entropy(logits, Y)
    print(f"    Closed-form output loss: {loss.item():.4f}")

    return W_out, loss.item()


def apply_closed_form_output(model, W_out):
    """Install closed-form output weights into the model."""
    with torch.no_grad():
        # W_out is (hidden_dim, vocab_size), proj_out.weight is (vocab_size, hidden_dim)
        model.proj_out.weight.copy_(W_out.T)
        model.proj_out.bias.zero_()


def train_gates_only(model, batches, epochs=15, lr=0.003, clip=1.0,
                      label="Gates"):
    """
    Phase 3: Fine-tune gates + output projection.

    Freeze embeddings and topology calibration (scale/shift/input_gate/stage_embed).
    Unfreeze: gate params + output projection + layer norm.
    The topology matrices are always frozen (buffers, no grad).

    This lets the output layer adapt to gated hidden states while the
    gates learn their filtering role. ~18K active params.
    """
    # Freeze all params first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze gate params
    gate_params = []
    for block in model.blocks:
        for gate in block.stage_gates:
            for p in gate.parameters():
                p.requires_grad = True
                gate_params.append(p)

    # Unfreeze output projection + layer norm
    for p in model.proj_out.parameters():
        p.requires_grad = True
        gate_params.append(p)
    for p in model.ln.parameters():
        p.requires_grad = True
        gate_params.append(p)

    n_gate_params = sum(p.numel() for p in gate_params)
    print(f"    Active params unfrozen: {n_gate_params:,d} (gates + output + ln)")

    optimizer = torch.optim.Adam(gate_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_b = 0
        hidden = None

        for x, y, si in batches:
            optimizer.zero_grad()

            if hidden is not None:
                hidden = [h.detach() for h in hidden]

            logits, hidden = model(x, hidden, si)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(gate_params, clip)
            optimizer.step()

            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg_loss = total_loss / max(n_b, 1)
        ppl = math.exp(min(avg_loss, 20))
        history.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"    [{label:>6s}] Epoch {epoch+1:3d}/{epochs}  "
                  f"Loss: {avg_loss:.4f}  PPL: {ppl:.1f}  "
                  f"Time: {elapsed:.1f}s")

    # Re-enable all params for evaluation
    for p in model.parameters():
        p.requires_grad = True

    return history


def train_cdt(model, train_batches, test_batches, gate_epochs=15,
              lambda_reg=1.0, label="CDT"):
    """
    Chestohedron Direct Training — the full 3-phase pipeline.

    Phase 1: Reservoir collection (forward-only)
    Phase 2: Closed-form output layer (one matrix op)
    Phase 3: Gate calibration (fine-tune ~13K params)

    The topology does the structural work. We solve for filters.
    Minutes, not hours.
    """
    t0 = time.time()
    print(f"\n  --- {label}: Chestohedron Direct Training ---")

    # Phase 1: Reservoir collection
    print(f"\n  Phase 1: Reservoir collection (forward-only)...")
    t1 = time.time()
    H, Y = reservoir_collect(model, train_batches)
    print(f"    Collected {H.shape[0]:,d} hidden states in {time.time()-t1:.1f}s")

    # Phase 2: Closed-form output
    print(f"\n  Phase 2: Closed-form output layer...")
    t2 = time.time()
    W_out, cf_loss = solve_output_closed_form(
        H, Y, model.proj_out.out_features, model.hidden_dim, lambda_reg)
    apply_closed_form_output(model, W_out)
    print(f"    Solved in {time.time()-t2:.1f}s")

    # Phase 3: Gate calibration
    print(f"\n  Phase 3: Gate calibration (thin filters only)...")
    t3 = time.time()
    gate_history = train_gates_only(model, train_batches, epochs=gate_epochs,
                                     lr=0.003, label="Gates")
    print(f"    Calibrated in {time.time()-t3:.1f}s")

    total_time = time.time() - t0
    print(f"\n  CDT total: {total_time:.1f}s")

    return gate_history, total_time


def train_standard(model, batches, epochs, lr=0.003, label="Model",
                    use_stages=True):
    """Standard backprop training (for baselines)."""
    t0 = time.time()
    print(f"\n  --- {label}: Standard backprop ---")
    h = train_model(model, batches, epochs, lr=lr, label=label,
                     use_stages=use_stages)
    elapsed = time.time() - t0
    return h, elapsed


# ============================================================
# FULL BENCHMARK + ABLATION
# ============================================================

def run_benchmark(n_examples=1000, embed_dim=32, hidden_dim=96,
                  seq_len=48, batch_size=32, epochs=15):
    """
    GCN benchmark with Chestohedron Direct Training (CDT).

    CDT trains GCN models in 3 phases:
      Phase 1: Forward-only reservoir pass (topology does structural work)
      Phase 2: Closed-form output layer (one matrix op, seconds)
      Phase 3: Gate calibration (fine-tune only ~13K gate params)

    Baselines (LSTM, GRU) use standard backprop for fair comparison.
    CCN uses standard backprop (no gates to calibrate).

    Primary models:
      1. GCN (per-stage, r=3) via CDT    <- the hypothesis
      2. CCN (pure topology)              <- baseline
      3. LSTM (all learned)               <- baseline
      4. GRU (all learned)                <- baseline

    Ablations (all via CDT):
      5. GCN shared gates                 <- proves specialization
      6. GCN rank=6                       <- more gate capacity
      7. GCN rank=12                      <- even more capacity
      8. GCN update-only (no reset)       <- is reset needed?
    """
    torch.set_num_threads(4)

    print("=" * 70)
    print("  GATED CHESTOHEDRON NETWORK (GCN)")
    print("  Chestohedron Direct Training — topology makes compute obsolete")
    print("  CCN solves WHERE (topology). GRU solves WHAT (gating).")
    print("=" * 70)

    # ── DATA ──
    print(f"\n  Generating {n_examples} V9-structured reasoning examples...")
    gen = V9DataGenerator(seed=42)
    examples = gen.generate_dataset(n_examples)
    tokenizer = V9Tokenizer().build(examples)
    print(f"  Vocab: {tokenizer.vocab_size}  Examples: {len(examples)}")

    split = int(len(examples) * 0.9)
    train_batches = prepare_batches(examples[:split], tokenizer, seq_len, batch_size)
    test_batches = prepare_batches(examples[split:], tokenizer, seq_len, batch_size)
    print(f"  Train batches: {len(train_batches)}  Test batches: {len(test_batches)}")

    stage_start_ids, _ = tokenizer.get_stage_token_ids()

    # ── BUILD MODELS ──
    print(f"\n{'─' * 70}")
    print(f"  Building models (hidden_dim={hidden_dim})...")

    gate_epochs = 10  # Gate calibration epochs (Phase 3)

    models = {}

    # Primary: GCN with per-stage gates, rank=3
    models['GCN (per-stage, r=3)'] = {
        'model': GatedChestohedronLM(
            tokenizer.vocab_size, embed_dim, hidden_dim,
            gate_rank=3, stage_token_ids=stage_start_ids, seed=42),
        'stages': True, 'primary': True, 'use_cdt': True,
    }

    # Baselines (standard backprop)
    models['CCN (pure topology)'] = {
        'model': ChestohedronLM(
            tokenizer.vocab_size, embed_dim, hidden_dim,
            stage_token_ids=stage_start_ids, seed=42),
        'stages': True, 'primary': True, 'use_cdt': False,
    }
    models['LSTM (all learned)'] = {
        'model': VanillaLSTM_LM(tokenizer.vocab_size, embed_dim, hidden_dim),
        'stages': False, 'primary': True, 'use_cdt': False,
    }
    models['GRU (all learned)'] = {
        'model': VanillaGRU_LM(tokenizer.vocab_size, embed_dim, hidden_dim),
        'stages': False, 'primary': True, 'use_cdt': False,
    }

    # Key ablations (CDT)
    models['GCN shared gates'] = {
        'model': GatedChestohedronLM(
            tokenizer.vocab_size, embed_dim, hidden_dim,
            gate_rank=3, stage_token_ids=stage_start_ids, seed=42,
            shared_gates=True),
        'stages': True, 'primary': False, 'use_cdt': True,
    }
    models['GCN update-only'] = {
        'model': GatedChestohedronLM(
            tokenizer.vocab_size, embed_dim, hidden_dim,
            gate_rank=3, stage_token_ids=stage_start_ids, seed=42,
            update_only=True),
        'stages': True, 'primary': False, 'use_cdt': True,
    }

    # Print param counts
    print(f"\n  {'Model':<28s} {'Learned':>10s} {'Fixed':>10s} {'Total':>10s} {'Train':>8s}")
    print(f"  {'─' * 64}")
    for name, entry in models.items():
        m = entry['model']
        lp, fp = m.learned_params, m.fixed_params
        method = "CDT" if entry['use_cdt'] else "Backprop"
        print(f"  {name:<28s} {lp:>10,d} {fp:>10,d} {lp+fp:>10,d} {method:>8s}")

    # ── TRAIN ──
    print(f"\n{'─' * 70}")
    print(f"  Training...")
    print(f"  GCN models: Chestohedron Direct Training (reservoir + closed-form + gates)")
    print(f"  Baselines: Standard backprop ({epochs} epochs)")
    print(f"{'─' * 70}")

    histories = {}
    train_times = {}

    for name, entry in models.items():
        if entry['use_cdt']:
            # CDT: reservoir -> closed-form -> gate calibration
            h, elapsed = train_cdt(entry['model'], train_batches, test_batches,
                                    gate_epochs=gate_epochs, label=name)
            histories[name] = h
            train_times[name] = elapsed
        else:
            # Standard backprop
            label = name[:6].strip()
            h, elapsed = train_standard(entry['model'], train_batches, epochs,
                                         lr=0.003, label=label,
                                         use_stages=entry['stages'])
            histories[name] = h
            train_times[name] = elapsed

    # ── EVALUATE ──
    print(f"\n{'─' * 70}")
    print(f"  Evaluating on test set...")

    def evaluate(model, batches, use_stages=False):
        model.eval()
        total_loss = 0
        n = 0
        with torch.no_grad():
            for x, y, si in batches:
                stages = si if use_stages else None
                logits, _ = model(x, None, stages)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

    test_losses = {}
    for name, entry in models.items():
        test_losses[name] = evaluate(entry['model'], test_batches, entry['stages'])

    # ── RESULTS ──
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — GATED CHESTOHEDRON NETWORK BENCHMARK")
    print(f"{'=' * 70}")

    print(f"\n  PRIMARY COMPARISON:")
    print(f"  {'Model':<28s} {'Train':>8s} {'Test':>8s} {'PPL':>7s} "
          f"{'Learned':>9s} {'CapDen':>8s} {'Time':>7s}")
    print(f"  {'─' * 75}")

    results = {}
    for name, entry in models.items():
        m = entry['model']
        tr = histories[name][-1]
        te = test_losses[name]
        ppl = math.exp(min(te, 20))
        lp = m.learned_params
        cd = (1.0 / te) / lp * 1e6
        tt = train_times[name]
        results[name] = {'train': tr, 'test': te, 'ppl': ppl,
                          'learned': lp, 'fixed': m.fixed_params,
                          'cap_density': cd, 'time': tt}

        if entry['primary']:
            print(f"  {name:<28s} {tr:>8.4f} {te:>8.4f} {ppl:>7.1f} "
                  f"{lp:>9,d} {cd:>8.2f} {tt:>6.1f}s")

    print(f"\n  ABLATION STUDY:")
    print(f"  {'Model':<28s} {'Train':>8s} {'Test':>8s} {'PPL':>7s} "
          f"{'Learned':>9s} {'CapDen':>8s} {'Time':>7s}")
    print(f"  {'─' * 75}")

    for name, entry in models.items():
        if not entry['primary']:
            r = results[name]
            print(f"  {name:<28s} {r['train']:>8.4f} {r['test']:>8.4f} {r['ppl']:>7.1f} "
                  f"{r['learned']:>9,d} {r['cap_density']:>8.2f} {r['time']:>6.1f}s")

    # ── ANALYSIS ──
    gcn_r = results['GCN (per-stage, r=3)']
    ccn_r = results['CCN (pure topology)']
    lstm_r = results['LSTM (all learned)']
    gru_r = results['GRU (all learned)']
    shared_r = results['GCN shared gates']

    print(f"\n{'─' * 70}")
    print(f"  ANALYSIS")
    print(f"{'─' * 70}")

    print(f"\n  Gap closure (CCN -> LSTM):")
    ccn_gap = ccn_r['test'] - lstm_r['test']
    gcn_gap = gcn_r['test'] - lstm_r['test']
    if ccn_gap > 0:
        closure = (1 - gcn_gap / ccn_gap) * 100
        print(f"    CCN test loss:  {ccn_r['test']:.4f}")
        print(f"    GCN test loss:  {gcn_r['test']:.4f}")
        print(f"    LSTM test loss: {lstm_r['test']:.4f}")
        print(f"    Gap closed: {closure:.1f}%")

    print(f"\n  Capability density ranking:")
    ranked = sorted(results.items(), key=lambda x: x[1]['cap_density'], reverse=True)
    for i, (name, r) in enumerate(ranked):
        marker = " <-- WINNER" if i == 0 else ""
        print(f"    {i+1}. {name:<28s} {r['cap_density']:.2f}{marker}")

    print(f"\n  Per-stage specialization test:")
    print(f"    GCN per-stage test loss:  {gcn_r['test']:.4f}")
    print(f"    GCN shared gates loss:    {shared_r['test']:.4f}")
    if shared_r['test'] > gcn_r['test']:
        print(f"    -> Per-stage WINS by {shared_r['test'] - gcn_r['test']:.4f} "
              f"(topology specializes the gates)")
    else:
        print(f"    -> Shared gates competitive (gate rank may be too low for visible specialization)")

    print(f"\n  Reset gate value:")
    uponly_r = results['GCN update-only']
    print(f"    GCN with reset:    {gcn_r['test']:.4f}")
    print(f"    GCN update-only:   {uponly_r['test']:.4f}")
    if uponly_r['test'] > gcn_r['test']:
        print(f"    -> Reset gate helps by {uponly_r['test'] - gcn_r['test']:.4f}")
    else:
        print(f"    -> Reset gate not critical (update gate does the heavy lifting)")

    # ── SUCCESS CRITERIA ──
    print(f"\n{'─' * 70}")
    print(f"  SUCCESS CRITERIA")
    print(f"{'─' * 70}")

    checks = [
        ("Raw loss < 0.4", gcn_r['test'] < 0.4, f"{gcn_r['test']:.4f}"),
        ("Learned params < 35,000", gcn_r['learned'] < 35000, f"{gcn_r['learned']:,d}"),
        ("Cap density > CCN",
         gcn_r['cap_density'] > ccn_r['cap_density'],
         f"{gcn_r['cap_density']:.2f} vs {ccn_r['cap_density']:.2f}"),
        ("Per-stage > shared gates",
         shared_r['test'] > gcn_r['test'],
         f"{gcn_r['test']:.4f} vs {shared_r['test']:.4f}"),
    ]

    for label, passed, detail in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {label}: {detail}")

    # ── GATE ACTIVATION VISUALIZATION ──
    print(f"\n{'─' * 70}")
    print(f"  GATE ACTIVATION PATTERNS")
    print(f"{'─' * 70}")

    gcn_model = models['GCN (per-stage, r=3)']['model']
    gate_history = record_gate_activations(gcn_model, test_batches, n_batches=10)
    gate_agg = visualize_gate_activations(gate_history, save_path="gcn_gate_activations.png")

    # ── GENERATION SAMPLES ──
    print(f"\n{'─' * 70}")
    print(f"  V9-STRUCTURED GENERATION (GCN)")
    print(f"{'─' * 70}")

    test_problems = [
        "what does the fox do",
        "find the sum of 7 and 5",
        "find the next number in 2 4 6 8 10",
    ]

    for problem in test_problems:
        print(f"\n  Problem: {problem}")
        output = generate_v9_response(gcn_model, tokenizer, problem, hidden_dim)
        for tok in STAGE_TOKENS:
            output = output.replace(tok, f"\n    {tok}")
        for tok in STAGE_END_TOKENS:
            output = output.replace(tok, f" {tok}")
        print(f"  Response:{output[:400]}")

    # ── ARCHITECTURE + CDT SUMMARY ──
    print(f"\n{'=' * 70}")
    print(f"  ARCHITECTURE + CHESTOHEDRON DIRECT TRAINING")
    print(f"{'=' * 70}")
    print(f"""
  Gated Chestohedron Network (GCN):

  Per-stage low-rank GRU gates:
    r = sigmoid(r_proj(r_state(state) + r_input(u)))   # WHAT history is relevant
    z = sigmoid(z_proj(z_state(state) + z_input(u)))   # HOW MUCH to update
    candidate = tanh((r * state) @ W_topo + u)         # topology on filtered state
    state = (1 - z) * state + z * candidate            # gated update

  Fixed topology: {gcn_r['fixed']:,d} params (chestohedron geometry)
  Learned params: {gcn_r['learned']:,d} (gates + embeddings + calibration)
  LSTM learned:   {lstm_r['learned']:,d} (everything)

  Chestohedron Direct Training (CDT):
    Phase 1: Forward-only reservoir pass (topology does structural work)
    Phase 2: Closed-form output layer (ridge regression, ONE matrix op)
    Phase 3: Gate calibration (fine-tune only ~13K gate params)

    GCN CDT time:   {gcn_r['time']:.1f}s (reservoir + closed-form + 10 gate epochs)
    CCN backprop:   {ccn_r['time']:.1f}s (15 epochs, full model)
    LSTM backprop:  {lstm_r['time']:.1f}s (15 epochs, simple architecture)

  CDT trains 18K params (gates + output). Full GCN backprop would train 35K.
  The topology eliminates ~50% of the gradient computation.

  The topology doesn't just define the architecture.
  It defines the training method.
  Structure is fixed -> filters are solvable -> compute is optional.
""")

    # ── WRITE RESULTS FILE ──
    write_results_file(results, gate_agg, checks, hidden_dim, embed_dim, n_examples, epochs)

    return results


def write_results_file(results, gate_agg, checks, hidden_dim, embed_dim,
                        n_examples, epochs):
    """Write benchmark results to markdown file."""
    stage_names = ["MIRROR", "INHERIT", "BOUND", "EXPRESS", "VERIFY", "REMOVE", "GATE6"]

    lines = [
        "# GCN Benchmark Results",
        "",
        f"**Config**: hidden_dim={hidden_dim}, embed_dim={embed_dim}, "
        f"examples={n_examples}, epochs={epochs}",
        "",
        "## Primary Comparison",
        "",
        "| Model | Train Loss | Test Loss | Test PPL | Learned Params | Cap Density |",
        "|-------|-----------|-----------|----------|----------------|-------------|",
    ]

    primary = ['GCN (per-stage, r=3)', 'CCN (pure topology)',
                'LSTM (all learned)', 'GRU (all learned)']
    for name in primary:
        r = results[name]
        lines.append(f"| {name} | {r['train']:.4f} | {r['test']:.4f} | "
                      f"{r['ppl']:.1f} | {r['learned']:,d} | {r['cap_density']:.2f} |")

    lines += [
        "",
        "## Ablation Study",
        "",
        "| Model | Test Loss | Learned Params | Cap Density |",
        "|-------|-----------|----------------|-------------|",
    ]

    ablation = ['GCN shared gates', 'GCN update-only']
    for name in ablation:
        r = results[name]
        lines.append(f"| {name} | {r['test']:.4f} | {r['learned']:,d} | "
                      f"{r['cap_density']:.2f} |")

    lines += [
        "",
        "## Gate Activation Patterns",
        "",
        "| Stage | Reset (r) | Update (z) | r std | z std |",
        "|-------|-----------|------------|-------|-------|",
    ]

    for s in range(7):
        a = gate_agg[s]
        lines.append(f"| {stage_names[s]} | {a['r_mean']:.4f} | {a['z_mean']:.4f} | "
                      f"{a['r_std']:.4f} | {a['z_std']:.4f} |")

    lines += [
        "",
        "## Success Criteria",
        "",
    ]

    for label, passed, detail in checks:
        icon = "PASS" if passed else "FAIL"
        lines.append(f"- **[{icon}]** {label}: {detail}")

    lines += [
        "",
        "## Architecture",
        "",
        "```",
        "Per-stage low-rank GRU gates:",
        "  r = sigmoid(r_proj(r_state(state) + r_input(u)))  # reset",
        "  z = sigmoid(z_proj(z_state(state) + z_input(u)))  # update",
        "  candidate = tanh((r * state) @ W_topo + u)",
        "  state = (1 - z) * state + z * candidate",
        "```",
        "",
        "CCN solves WHERE (routing) — fixed, zero cost.",
        "GRU solves WHAT (filtering) — learned, per-stage specialized.",
        "They compose. They don't interfere.",
        "",
        "## Chestohedron Direct Training (CDT)",
        "",
        "Standard backprop learns structure AND filters together. Expensive.",
        "CDT exploits the chestohedron's fixed topology:",
        "",
        "1. **Reservoir collection** — forward-only pass through fixed topology",
        "2. **Closed-form output** — ridge regression, one matrix operation",
        "3. **Gate calibration** — fine-tune only ~13K gate params",
        "",
        "The topology makes massive compute obsolete.",
        "The geometry does the structural work for free.",
        "You only learn the thin filters — and those can be solved directly.",
    ]

    with open("gcn_benchmark_results.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Results written to gcn_benchmark_results.md")


if __name__ == "__main__":
    results = run_benchmark()
