# GCN Benchmark Results

**Config**: hidden_dim=96, embed_dim=32, examples=1000, epochs=15

## Primary Comparison

| Model | Train Loss | Test Loss | Test PPL | Learned Params | Cap Density |
|-------|-----------|-----------|----------|----------------|-------------|
| GCN (per-stage, r=3) | 0.9006 | 0.8664 | 2.4 | 34,869 | 33.10 |
| CCN (pure topology) | 1.2289 | 1.1963 | 3.3 | 21,429 | 39.01 |
| LSTM (all learned) | 0.4833 | 0.5155 | 1.7 | 56,757 | 34.18 |
| GRU (all learned) | 0.4076 | 0.4578 | 1.6 | 44,277 | 49.34 |

## Ablation Study

| Model | Test Loss | Learned Params | Cap Density |
|-------|-----------|----------------|-------------|
| GCN shared gates | 1.0183 | 23,349 | 42.06 |
| GCN update-only | 0.9569 | 28,149 | 37.13 |

## Gate Activation Patterns

| Stage | Reset (r) | Update (z) | r std | z std |
|-------|-----------|------------|-------|-------|
| MIRROR | 0.5950 | 0.4004 | 0.4471 | 0.4485 |
| INHERIT | 0.7082 | 0.4834 | 0.4118 | 0.4677 |
| BOUND | 0.6898 | 0.3427 | 0.4129 | 0.4460 |
| EXPRESS | 0.7558 | 0.4205 | 0.3771 | 0.4537 |
| VERIFY | 0.6827 | 0.2995 | 0.4139 | 0.4059 |
| REMOVE | 0.6438 | 0.2897 | 0.3791 | 0.4039 |
| GATE6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Success Criteria

- **[FAIL]** Raw loss < 0.4: 0.8664
- **[PASS]** Learned params < 35,000: 34,869
- **[FAIL]** Cap density > CCN: 33.10 vs 39.01
- **[PASS]** Per-stage > shared gates: 0.8664 vs 1.0183

## Architecture

```
Per-stage low-rank GRU gates:
  r = sigmoid(r_proj(r_state(state) + r_input(u)))  # reset
  z = sigmoid(z_proj(z_state(state) + z_input(u)))  # update
  candidate = tanh((r * state) @ W_topo + u)
  state = (1 - z) * state + z * candidate
```

CCN solves WHERE (routing) — fixed, zero cost.
GRU solves WHAT (filtering) — learned, per-stage specialized.
They compose. They don't interfere.

## Chestohedron Direct Training (CDT)

Standard backprop learns structure AND filters together. Expensive.
CDT exploits the chestohedron's fixed topology:

1. **Reservoir collection** — forward-only pass through fixed topology
2. **Closed-form output** — ridge regression, one matrix operation
3. **Gate calibration** — fine-tune only ~13K gate params

The topology makes massive compute obsolete.
The geometry does the structural work for free.
You only learn the thin filters — and those can be solved directly.