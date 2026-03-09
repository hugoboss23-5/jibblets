"""
TOPOLOGY OVER SUBSTANCE — DEFINITIVE BENCHMARK
================================================
7 architectures × 2 datasets × 10 seeds = 140 core runs

Architectures:
  1. CCN          — chestohedron topology (the thesis)
  2. GCN-per      — chestohedron + per-stage GRU gates
  3. GCN-shared   — chestohedron + shared gates (ablation)
  4. CCN-scrambled — same matrices, random stage assignment
  5. Random-ESN   — random fixed matrices (not geometric)
  6. LSTM         — all-learned baseline
  7. GRU          — all-learned baseline

Datasets:
  V9-structured  — topology's home turf
  Shakespeare    — neutral ground, no architecture advantage

Every run: same hidden_dim, embed_dim, epochs, lr, optimizer, scheduler.
10 seeds with mean ± std. This is publication-grade methodology.
"""

import sys, os, json, time, math, csv, random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from chestohedron_lm import (
    V9DataGenerator, V9Tokenizer, ChestohedronLM, ChestohedronBlock,
    VanillaLSTM_LM, VanillaGRU_LM, make_topology_matrices,
    prepare_batches, train_model, STAGE_TOKENS, STAGE_END_TOKENS, PHI,
)
from chestohedron_gcn import GatedChestohedronLM

# ============================================================
# NEW: SCRAMBLED TOPOLOGY — same matrices, random stage assignment
# Proves whether SPECIFIC geometry-to-function mapping matters
# ============================================================

class ScrambledChestohedronBlock(ChestohedronBlock):
    """Same 7 geometric matrices as CCN, but randomly assigned to stages.
    If ordered ≈ scrambled → specific mapping is decorative.
    If ordered > scrambled → geometry-to-function mapping carries signal."""

    def __init__(self, dim, seed=42, scramble_seed=999):
        super().__init__(dim, seed)
        rng = random.Random(scramble_seed)
        self._scramble_order = list(range(6))
        rng.shuffle(self._scramble_order)

    def _get_stage_matrices(self):
        original = super()._get_stage_matrices()
        return [original[self._scramble_order[i]] for i in range(6)]


class ScrambledChestohedronLM(ChestohedronLM):
    """Language model with scrambled topology assignment."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_cycles=1,
                 stage_token_ids=None, seed=42, scramble_seed=999):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_cycles,
                         stage_token_ids, seed)
        self.blocks = nn.ModuleList([
            ScrambledChestohedronBlock(hidden_dim, seed=seed + i,
                                       scramble_seed=scramble_seed)
            for i in range(n_cycles)
        ])


# ============================================================
# NEW: RANDOM RESERVOIR LM — tests if ANY fixed topology helps
# 7 random matrices instead of geometrically specific ones
# ============================================================

class RandomReservoirBlock(nn.Module):
    """7 random spectral-scaled matrices. No geometric structure.
    If CCN > Random → chestohedron geometry specifically helps.
    If CCN ≈ Random → any fixed topology works equally."""

    def __init__(self, dim, seed=42):
        super().__init__()
        self.dim = dim
        torch.manual_seed(seed)
        for i in range(7):
            W = torch.randn(dim, dim) / math.sqrt(dim)
            eigs = torch.linalg.eigvals(W).abs()
            sr = eigs.max().item()
            if sr > 1e-8:
                W = W * (0.9 / sr)
            self.register_buffer(f'W_{i}', W)
        self.stage_scale = nn.ParameterList([
            nn.Parameter(torch.ones(dim)) for _ in range(7)])
        self.stage_shift = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(7)])
        self.input_gate = nn.Linear(dim, dim, bias=False)
        self.stage_embed = nn.Embedding(7, dim)

    def forward(self, state, x, stage_idx=None):
        u = self.input_gate(x)
        if stage_idx is not None:
            u = u + self.stage_embed(stage_idx) * 0.3
        for i in range(7):
            W = getattr(self, f'W_{i}')
            h = state @ W + u
            h = h * self.stage_scale[i] + self.stage_shift[i]
            state = torch.tanh(h)
        return state


class RandomReservoirLM(nn.Module):
    """Language model with random fixed topology (ESN-style)."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_cycles=1,
                 stage_token_ids=None, seed=42):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = n_cycles
        self.stage_token_map = {}
        if stage_token_ids:
            for i, tid in enumerate(stage_token_ids):
                self.stage_token_map[tid] = i
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj_in = nn.Linear(embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            RandomReservoirBlock(hidden_dim, seed=seed + i)
            for i in range(n_cycles)])
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
        return self.proj_out(out), hidden

    @property
    def learned_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def fixed_params(self):
        return sum(b.numel() for b in self.buffers())


# ============================================================
# SHAKESPEARE DATA PIPELINE — neutral ground, no V9 structure
# ============================================================

def prepare_shakespeare_batches(filepath, seq_len=96, batch_size=16,
                                 train_frac=0.9):
    """Character-level Shakespeare. No stage markers. Pure language."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    tokens = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    split = int(len(tokens) * train_frac)

    def make_batches(toks):
        n = len(toks) - 1
        usable = (n // (seq_len * batch_size)) * seq_len * batch_size
        if usable == 0:
            return []
        inputs = toks[:usable].reshape(-1, seq_len)
        targets = toks[1:usable + 1].reshape(-1, seq_len)
        stages = torch.zeros_like(inputs)  # no V9 stages
        batches = []
        n_batches = inputs.shape[0] // batch_size
        for i in range(n_batches):
            s = i * batch_size
            batches.append((inputs[s:s+batch_size], targets[s:s+batch_size],
                            stages[s:s+batch_size]))
        return batches

    return (make_batches(tokens[:split]), make_batches(tokens[split:]),
            vocab_size)


def evaluate(model, batches, use_stages=False, device='cpu'):
    """Evaluate model on test batches."""
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for x, y, si in batches:
            x, y, si = x.to(device), y.to(device), si.to(device)
            stages = si if use_stages else None
            logits, _ = model(x, None, stages)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    y.reshape(-1))
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


# ============================================================
# CONFIGURATION — locked across all runs
# ============================================================

CONFIG = {
    'hidden_dim': 96, 'embed_dim': 32, 'epochs': 30, 'lr': 0.003,
    'batch_size': 16, 'seq_len': 96, 'clip': 1.0, 'n_examples': 3000,
    'gate_rank': 3,
    'seeds': [42, 123, 456, 789, 1337, 2024, 3141, 4269, 5555, 7777],
}

QUICK_CONFIG = {
    **CONFIG, 'epochs': 5, 'n_examples': 1000, 'seeds': [42, 123],
}

ARCHITECTURES = ['ccn', 'gcn_per', 'gcn_shared', 'scrambled',
                 'random_reservoir', 'lstm', 'gru']
DATASETS = ['v9', 'shakespeare']


def build_model(arch, vocab_size, embed_dim, hidden_dim,
                stage_token_ids, seed, gate_rank=3):
    """Factory for all 7 architectures."""
    if arch == 'ccn':
        return ChestohedronLM(vocab_size, embed_dim, hidden_dim, 1,
                               stage_token_ids, seed)
    elif arch == 'gcn_per':
        return GatedChestohedronLM(vocab_size, embed_dim, hidden_dim, 1,
                                    gate_rank, stage_token_ids, seed,
                                    shared_gates=False)
    elif arch == 'gcn_shared':
        return GatedChestohedronLM(vocab_size, embed_dim, hidden_dim, 1,
                                    gate_rank, stage_token_ids, seed,
                                    shared_gates=True)
    elif arch == 'scrambled':
        return ScrambledChestohedronLM(vocab_size, embed_dim, hidden_dim, 1,
                                        stage_token_ids, seed,
                                        scramble_seed=seed + 10000)
    elif arch == 'random_reservoir':
        return RandomReservoirLM(vocab_size, embed_dim, hidden_dim, 1,
                                  stage_token_ids, seed)
    elif arch == 'lstm':
        return VanillaLSTM_LM(vocab_size, embed_dim, hidden_dim)
    elif arch == 'gru':
        return VanillaGRU_LM(vocab_size, embed_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown: {arch}")


def arch_uses_stages(arch):
    return arch in ('ccn', 'gcn_per', 'gcn_shared', 'scrambled',
                    'random_reservoir')


# ============================================================
# SINGLE RUN — one arch × one dataset × one seed
# ============================================================

def run_single(arch, dataset, seed, cfg, device='cpu'):
    """Execute one training run. Returns result dict."""
    t0 = time.time()

    if dataset == 'v9':
        gen = V9DataGenerator(seed=seed)
        examples = gen.generate_dataset(cfg['n_examples'])
        tokenizer = V9Tokenizer().build(examples)
        vocab_size = tokenizer.vocab_size
        split = int(len(examples) * 0.9)
        train_b = prepare_batches(examples[:split], tokenizer,
                                   cfg['seq_len'], cfg['batch_size'])
        test_b = prepare_batches(examples[split:], tokenizer,
                                  cfg['seq_len'], cfg['batch_size'])
        stage_ids, _ = tokenizer.get_stage_token_ids()
    elif dataset == 'shakespeare':
        train_b, test_b, vocab_size = prepare_shakespeare_batches(
            'shakespeare.txt', cfg['seq_len'], cfg['batch_size'])
        stage_ids = None

    # Move to device
    train_b = [(x.to(device), y.to(device), s.to(device))
                for x, y, s in train_b]
    test_b = [(x.to(device), y.to(device), s.to(device))
               for x, y, s in test_b]

    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(arch, vocab_size, cfg['embed_dim'],
                         cfg['hidden_dim'], stage_ids, seed,
                         cfg.get('gate_rank', 3)).to(device)

    use_stg = arch_uses_stages(arch) and dataset == 'v9'
    label = f"{arch[:8]:>8s}/{dataset[:5]:>5s}/s{seed}"
    train_model(model, train_b, cfg['epochs'], cfg['lr'], cfg['clip'],
                label=label, use_stages=use_stg)

    test_loss = evaluate(model, test_b, use_stg, device)
    learned = model.learned_params
    fixed = getattr(model, 'fixed_params', 0)
    elapsed = time.time() - t0
    density = (1.0 / test_loss) / learned * 1e6 if test_loss > 0 else 0

    print(f"  >> {arch:>15s} | {dataset:>11s} | seed={seed:>5d} | "
          f"loss={test_loss:.4f} params={learned:>6d} "
          f"density={density:.2f} ({elapsed:.0f}s)")

    return {'arch': arch, 'dataset': dataset, 'seed': seed,
            'test_loss': test_loss, 'test_ppl': math.exp(min(test_loss, 20)),
            'learned_params': learned, 'fixed_params': fixed,
            'cap_density': density, 'time_s': elapsed}


# ============================================================
# FULL BENCHMARK — 7 × 2 × 10 factorial
# ============================================================

def run_full_benchmark(quick=False):
    cfg = QUICK_CONFIG if quick else CONFIG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 75)
    print("  TOPOLOGY OVER SUBSTANCE — DEFINITIVE BENCHMARK")
    print("=" * 75)
    print(f"  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")
    print(f"  Config: h={cfg['hidden_dim']} e={cfg['embed_dim']} "
          f"ep={cfg['epochs']} seeds={len(cfg['seeds'])}")
    print(f"  Mode: {'QUICK' if quick else 'FULL'}")

    total = len(ARCHITECTURES) * len(DATASETS) * len(cfg['seeds'])
    print(f"  Total runs: {total}")
    print(f"  Archs: {', '.join(ARCHITECTURES)}")
    print()

    all_results = []
    run_n = 0
    t_start = time.time()

    for dataset in DATASETS:
        print(f"\n{'─' * 75}")
        print(f"  DATASET: {dataset.upper()}")
        print(f"{'─' * 75}")
        for arch in ARCHITECTURES:
            for seed in cfg['seeds']:
                run_n += 1
                print(f"\n  Run {run_n}/{total}:")
                try:
                    r = run_single(arch, dataset, seed, cfg, device)
                    all_results.append(r)
                except Exception as e:
                    print(f"  ERROR: {arch}/{dataset}/s{seed}: {e}")
                    import traceback; traceback.print_exc()
                    all_results.append({'arch': arch, 'dataset': dataset,
                                         'seed': seed, 'error': str(e)})

    total_time = time.time() - t_start
    print(f"\n\n  Total time: {total_time/60:.1f} minutes")


    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    agg = defaultdict(list)
    for r in all_results:
        if 'error' not in r:
            agg[(r['arch'], r['dataset'])].append(r)

    print(f"\n{'=' * 75}")
    print(f"  RESULTS")
    print(f"{'=' * 75}")

    for dataset in DATASETS:
        print(f"\n  === {dataset.upper()} ===")
        print(f"  {'Arch':<18s} {'Loss':>9s} {'±':>8s} {'Params':>7s} "
              f"{'Density':>9s} {'±':>8s} {'N':>3s}")
        print(f"  {'─' * 65}")
        for arch in ARCHITECTURES:
            runs = agg.get((arch, dataset), [])
            if not runs:
                print(f"  {arch:<18s} NO DATA")
                continue
            losses = [r['test_loss'] for r in runs]
            dens = [r['cap_density'] for r in runs]
            print(f"  {arch:<18s} {np.mean(losses):>9.4f} {np.std(losses):>8.4f} "
                  f"{runs[0]['learned_params']:>7d} {np.mean(dens):>9.2f} "
                  f"{np.std(dens):>8.2f} {len(runs):>3d}")

    # ============================================================
    # HYPOTHESIS TESTS
    # ============================================================
    def compare(a1, a2, ds, metric='test_loss'):
        r1 = [r[metric] for r in agg.get((a1, ds), [])]
        r2 = [r[metric] for r in agg.get((a2, ds), [])]
        if not r1 or not r2:
            return None
        diff = np.mean(r1) - np.mean(r2)
        se = math.sqrt(np.var(r1)/len(r1) + np.var(r2)/len(r2))
        t = diff / se if se > 0 else 0
        return {'diff': diff, 't': t, 'm1': np.mean(r1), 'm2': np.mean(r2)}

    tests = [
        ("H1: CCN > RandomESN (geometry helps)",
         'random_reservoir', 'ccn'),
        ("H2: CCN-ordered > CCN-scrambled (mapping matters)",
         'scrambled', 'ccn'),
        ("H3: GCN-per > GCN-shared (topology specializes gates)",
         'gcn_shared', 'gcn_per'),
        ("H4: CCN vs LSTM (topology replaces params)",
         'ccn', 'lstm'),
        ("H5: GCN vs LSTM (gates+topology vs pure learned)",
         'gcn_per', 'lstm'),
    ]


    print(f"\n{'=' * 75}")
    print(f"  HYPOTHESIS TESTS")
    print(f"{'=' * 75}")

    for ds in DATASETS:
        print(f"\n  --- {ds.upper()} ---")
        for desc, worse, better in tests:
            c = compare(worse, better, ds)
            if c:
                # Positive diff = worse has higher loss = better wins
                verdict = "SUPPORTED" if c['diff'] > 0 else "REJECTED"
                sig = "p<.05" if abs(c['t']) > 2 else "n.s."
                print(f"\n  {desc}")
                print(f"    {better}: {c['m2']:.4f}  vs  {worse}: {c['m1']:.4f}")
                print(f"    Δ={c['diff']:.4f}  t={c['t']:.2f}  [{verdict}] [{sig}]")

    # ============================================================
    # SAVE EVERYTHING
    # ============================================================
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    with open('benchmark_results.csv', 'w', newline='') as f:
        good = [r for r in all_results if 'error' not in r]
        if good:
            w = csv.DictWriter(f, fieldnames=good[0].keys())
            w.writeheader()
            w.writerows(good)

    with open('BENCHMARK_RESULTS.md', 'w') as f:
        f.write("# Topology Over Substance — Definitive Benchmark\n\n")
        f.write(f"**Config**: h={cfg['hidden_dim']}, e={cfg['embed_dim']}, "
                f"ep={cfg['epochs']}, seeds={len(cfg['seeds'])}, "
                f"device={device}\n\n")
        f.write(f"**Total time**: {total_time/60:.1f} minutes\n\n")
        for ds in DATASETS:
            f.write(f"## {ds.upper()}\n\n")
            f.write("| Arch | Loss Mean | Loss Std | Params | "
                    "Density Mean | Density Std | N |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for arch in ARCHITECTURES:
                runs = agg.get((arch, ds), [])
                if runs:
                    ls = [r['test_loss'] for r in runs]
                    ds2 = [r['cap_density'] for r in runs]
                    f.write(f"| {arch} | {np.mean(ls):.4f} | "
                            f"{np.std(ls):.4f} | {runs[0]['learned_params']} "
                            f"| {np.mean(ds2):.2f} | {np.std(ds2):.2f} "
                            f"| {len(runs)} |\n")
            f.write("\n")


        f.write("## Hypothesis Tests\n\n")
        for ds in DATASETS:
            f.write(f"### {ds.upper()}\n\n")
            for desc, worse, better in tests:
                c = compare(worse, better, ds)
                if c:
                    verdict = "SUPPORTED" if c['diff'] > 0 else "REJECTED"
                    f.write(f"**{desc}**\n")
                    f.write(f"- {better}: {c['m2']:.4f} vs "
                            f"{worse}: {c['m1']:.4f}\n")
                    f.write(f"- Δ={c['diff']:.4f}, t={c['t']:.2f} "
                            f"[{verdict}]\n\n")

    print(f"\n  Saved: benchmark_results.json, .csv, BENCHMARK_RESULTS.md")
    print(f"  Total wall time: {total_time/60:.1f} minutes")
    return all_results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    quick = '--quick' in sys.argv
    print(f"\n  Starting {'QUICK' if quick else 'FULL'} benchmark...")
    print(f"  Use --quick for sanity check (2 seeds, 5 epochs)\n")
    run_full_benchmark(quick=quick)
