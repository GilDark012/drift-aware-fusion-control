# Method — formal specification (equations & hyperparameters)

Self-contained formal statement of the mechanism, ready to paste into Chapter 2. Every
symbol maps to code and every hyperparameter to its config file. Values are those of the
**fixed-backbone re-run** (`artifacts/models/baseline-subsample/meta.json`,
`configs/detector_streaming.json`, `configs/adaptation.json`).

Notation: a user `u` has a chronological stream of items `x_1, x_2, …`; `e_i ∈ ℝ^d` is
the embedding of item `i` (shared table, `d = 128`); `b_i` its scalar bias.

## 1. Backbone — long/short representations and fusion

**Short-term** `S_u(t)` — a single-layer GRU over the last `W = 10` items, taking the
final hidden state (`fusion.py:short_term`):

```
S_u(t) = GRU( e_{x_{t-W}}, …, e_{x_{t-1}} )        ∈ ℝ^d
```

**Long-term** `L_u(t)` — EWMA-pooled embedding over up to `H = 50` preceding items with
half-life `h_L = 20` (`fusion.py:long_term`, `long_term_mode = "ewma"`):

```
L_u(t) = Σ_{m}  w_m · e_{x_m},     w_m ∝ 2^{-(t-1-m)/h_L},   Σ_m w_m = 1
```

**Fusion** with weight `α(t) ∈ [0,1]` (`fusion.py:fuse`) — `α=1` is pure long-term,
`α=0` pure short-term:

```
h_u(t) = α(t) · L_u(t) + (1 − α(t)) · S_u(t)
```

**Scoring** over the full catalogue:

```
score_u(t, i) = h_u(t) · e_i + b_i
```

The backbone is trained **once** and frozen; every baseline and the proposed system
differ only in how `α(t)` is set at inference (ADR-0002).

**Training** (`configs`/run meta): sampled-softmax over `n_neg = 100` uniform negatives,
cross-entropy on the true next item, `α` fixed at `0.5` during training, Adam
`lr = 10⁻³`, `weight_decay = 0`, `batch = 512`, `15` epochs, `dropout = 0.1`, seed `42`.
Preprocessing: **k-core `k = 5`** (ADR-0006).

## 2. Drift signal `D_u(t)`

**Primary — short-term self-divergence** (`signal.py:ShortTermDrift`, `h_D = 30`). Let
`S̄_u(t)` be the causal EWMA of the short-term state, *excluding* the current step:

```
S̄_u(t) = EWMA_{h_D}( S_u(1..t−1) )
D_u(t)  = 1 − cos( S_u(t), S̄_u(t) ),        D_u(1) = 0
```

Both terms live in the GRU output space, so a stable user has `D≈0` and a genuine
short-term shift raises `D` and holds it up until the slow average catches up. This is
the long/short divergence the project actually studies, expressed where the two terms
are geometrically comparable.

**Ablation — true long–short divergence** (`signal.py:LongShortDivergence`):

```
D^{LS}_u(t) = 1 − cos( L_u(t), S_u(t) )
```

Retained as an ablation. Empirically insensitive (stable-mean 0.84, category-change lift
1.04) because `L` (pooled embedding) and `S` (GRU state) occupy different geometric
subspaces — this is *why* the primary signal is used, and is reported as evidence, not
hidden (see `docs/rerun_fixed_backbone.md`).

## 3. Detector — causal z-score + persistence state machine

**Causal z-score** against the user's own trailing baseline (`rolling.py`,
`threshold.py:AdaptiveThreshold`), window `W_z = 10`, `min_periods = 3`:

```
μ_u(t) = mean( D_u(t−W_z+1 .. t) )
σ_u(t) = std ( D_u(t−W_z+1 .. t) )
z_u(t) = ( D_u(t) − μ_u(t) ) / ( σ_u(t) + ε )
```

The z-score makes the threshold mean the same thing for calm and volatile users
(hypothesis H6: per-user adaptive threshold vs a single global one).

**State machine** over `z_u(t)` (`detector.py`), states
`STABLE → EMERGING → CONFIRMED → STABLE`, with a `TEMPORARY` branch for rises that fade.
With enter level `z⁺ = 2.0`, exit level `z⁻ = 0.8`, persistence `p = 3`, gap tolerance
`g = 1`, recovery persistence `r = 3`, warm-up `8` steps forced STABLE (first `warmup`):

- **STABLE**: if `z_u(t) ≥ z⁺` → **EMERGING**, `streak = 1`, `onset = t`.
- **EMERGING**: if `z_u(t) ≥ z⁺` → `streak += 1`; when `streak ≥ p` → **CONFIRMED**,
  emit a shift `(onset, detect = t)`. Otherwise `gap += 1`; if `gap > g` → **TEMPORARY**
  (counted as a temporary deviation, never a confirmed shift).
- **CONFIRMED**: if `z_u(t) ≤ z⁻` → `recover += 1`; when `recover ≥ r` → **STABLE**.

Only a *persistent* rise (`p` elevated steps) confirms drift; a single spike never
triggers adaptation. The detection latency is `detect − onset`.

## 4. Fusion-weight policies (baselines + proposed)

All policies map the causal `(state, z)` series to `α(t)` (`adaptation/policy.py`):

**Static — Baseline A** (no drift awareness): `α(t) = 0.5`.

**window-B** (pure short-term, the trivial sliding window): `α(t) = 0`.

**Continuous — Baseline D** (adaptive `α` *without* a detector): with ceiling
`α_hi = 0.8`, sensitivity `λ = 0.1`, floor `α_lo = 0.2`,

```
α(t) = clip( α_hi − λ · max(z_u(t), 0),  α_lo,  α_hi )
```

**Proposed — state-driven, gradual (H4).** Per-state target `α*(s)`:
`STABLE = 0.8`, `EMERGING = 0.5`, `CONFIRMED = 0.2` (`TEMPORARY` behaves like STABLE).
The weight moves toward its target rate-limited by `Δ = 0.05` per step, from
`α(0) = α*(STABLE) = 0.8`:

```
α(t) = α(t−1) + clip( α*(s_t) − α(t−1),  −Δ,  +Δ )
```

So on a confirmed shift `α` glides `0.8 → 0.2` over ~12 steps (leaning on recent
behaviour), and glides back to `0.8` when the detector recovers.

## 5. Hyperparameter table (all values)

| Group | Symbol | Meaning | Value |
|---|---|---|---|
| Backbone | `d` | embedding = GRU hidden dim | 128 |
| | `W` | short window (GRU) | 10 |
| | `H`, `h_L` | long history, EWMA half-life | 50, 20 |
| | — | GRU layers, dropout | 1, 0.1 |
| | `min_context` | min prior items to score | 2 |
| Training | — | negatives (sampled-softmax) | 100, uniform |
| | — | α during training | fixed 0.5 |
| | — | lr, weight decay, batch, epochs | 10⁻³, 0, 512, 15 |
| | `k` | k-core preprocessing | 5 |
| Signal | `h_D` | short-term slow-EWMA half-life | 30 |
| Detector | `W_z`, min_periods | z-score window | 10, 3 |
| | `z⁺` / `z⁻` | enter / exit z levels | 2.0 / 0.8 |
| | `p` / `r` | persistence / recovery persistence | 3 / 3 |
| | `g` / warm-up | gap tolerance / warm-up | 1 / 8 |
| Policy | `α*` | STABLE / EMERGING / CONFIRMED | 0.8 / 0.5 / 0.2 |
| | `Δ` | max α move per step (gradual) | 0.05 |
| | `α_hi,λ,α_lo` | continuous-D ceiling/sens/floor | 0.8 / 0.1 / 0.2 |
| Eval | `k` | top-k | 10 |

Source files: `configs/detector_streaming.json`, `configs/adaptation.json`,
`artifacts/models/baseline-subsample/meta.json`.
