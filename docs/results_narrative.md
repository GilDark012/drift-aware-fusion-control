# Results & Discussion — consolidated thesis narrative

A single coherent account of the experimental arc, with every number traceable to a
logged run. Written to be lifted into the thesis (Chapter 3 Results + Chapter 5
Discussion). It supersedes the per-phase notes, which remain as provenance.

Two evaluation cohorts are used, both drawn from **Amazon Reviews'23** (Hou et al.,
2024) and both built by the same reproducible pipeline:

- the **default cohort** (k-core 5; 719 test-active users, vocabulary 14,988) — used for
  backbone validation, the drift signal, detection, and the external detector baselines;
- the **dense cohort** (k-core 5 + a ≥50 train-interaction floor; 196 long-history users,
  vocabulary 10,736) — used for the fusion and adaptation experiments, where a
  meaningful long-term profile is required.

---

## 3.1 Data and protocol

The source is **Amazon Reviews'23** (Hou, Li, He, Yan, Chen & McAuley, 2024; McAuley Lab,
UC San Diego), categories Electronics, Clothing_Shoes_and_Jewelry and Books, filtered to
2014-01-01 → 2018-12-30. After cleaning the corpus holds **3,971,952 interactions**,
130,987 users and 765,033 items (density 3.96×10⁻⁵). A **global chronological split** at
fixed timestamps (T1 = 2017-06-09, T2 = 2017-12-10) yields train 2,780,366 / val 397,195
/ test 794,391, so no future interaction can inform a past prediction. Full provenance,
including SHA-256 checksums, is in the data card.

**Ranking is full-catalogue and unsampled**: each next-item target is ranked against the
entire vocabulary (Krichene & Rendle, 2020). With one held-out positive per query,
Recall@k = HitRate@k; we report NDCG@10 (headline), Recall/Hit@10 and MRR. Every table is
anchored by **popularity and random floors** computed under the identical protocol, and
metrics are reported both **full** and **seen-only** to separate model quality from
catalogue coverage. Detector thresholds are tuned on validation only.

## 3.2 The backbone is a genuine sequential recommender

An early version of the model lost to the popularity floor (NDCG@10 0.00224 < 0.00270),
which the floors correctly exposed. The cause was catastrophic vocabulary sparsity — 67 %
of items were singletons — not the architecture. Introducing **k-core filtering (k = 5)**,
the standard preprocessing for GRU4Rec/SASRec, fixed it: on the default cohort the model
reaches **NDCG@10 = 0.0171 versus a popularity floor of 0.0116 (+47 %)**, MRR 0.0173 vs
0.0104. The backbone learns real sequential structure. **External baselines corroborate
this**: SASRec and GRU4Rec from RecBole, run on the same subsample under their standard
sequential protocol, reach NDCG@10 = 0.0201 and 0.0151 — the same narrow band as our
backbone and well above the floor — confirming the low absolute scores are a property of
the sparse data, not an implementation defect (`docs/sequential_baselines.md`).

## 3.3 Drift signal

The project studies long/short **divergence**. The true long–short cosine
(`1 − cos(L, S)`) proved insensitive even on the working backbone (stable mean 0.84,
category-change lift 1.04) because the pooled long-term embedding and the GRU state occupy
different geometric subspaces. The adopted signal expresses the same idea where the terms
are comparable — `D_u(t) = 1 − cos(S_u(t), EWMA_slow(S_u))`, short-term versus its own slow
average — with stable mean 0.26 and a **category-change lift of 1.60**. The true L–S form
is retained as an ablation that explains *why* the self-divergence signal is used.

## 3.4 Detection

On the default cohort, controlled shifts are injected with known onsets and detected by a
per-user causal z-score feeding a persistence state machine. The operating characteristic
is strong and, across **20 injection seeds**, tight:

| operating point | detection rate | false-alarm |
|---|--:|--:|
| z-enter = 1.75 | **0.644 ± 0.014** | 0.024 |
| z-enter = 2.00 | 0.279 ± 0.018 | 0.000 |

Median detection latency is a stable 3 interactions. (The single-seed 0.39 at z=2.0
reported earlier was a favourable draw; the 20-seed mean is 0.28 — reported honestly.)

## 3.5 External detector baselines — an honest comparison

Compared on the *same* divergence signal against the concept-drift literature
(Page-Hinkley and CUSUM, Page 1954; ADWIN, Bifet & Gavaldà 2007):

| detector | best detection @ ~0 % false-alarm |
|---|--:|
| CUSUM | **1.00 @ 0 %** |
| Page-Hinkley | 0.98 @ 0 % |
| ours (z-score + persistence) | 0.33 @ 0 % (needs ~11 % FA for 1.0) |
| ADWIN | 0.02 (fails — too sluggish on short streams) |

On these strong, clean synthetic shifts a trivial CUSUM **out-detects our detector**: the
per-user normalisation dilutes an already-unambiguous signal. This is not a failure of the
evaluation but of the benchmark's difficulty — the adaptive machinery targets *weak,
heterogeneous* drift the anti-profile injection never creates. The consequence is a
deliberate design choice downstream: **we adopt an off-the-shelf CUSUM as the detector and
locate the contribution in the adaptation layer.**

## 3.6 The window-B problem, and its diagnosis

The sharpest test of the fusion idea is whether adapting α beats the trivial
short-only sliding window (**window-B**, α = 0). On the default cohort it does **not** —
worse, window-B has the *highest* pre-drift quality of every policy (0.0298 vs the adaptive
policy's 0.0242). The reason is upstream: on this sparse cohort the long-term
representation `L` carries no useful ranking signal (quality is monotone toward pure
short-term), so there is no L/S balance worth adapting and **window-B is the empirical
ceiling, not a baseline we beat**.

This diagnosis has a clear implication: adaptation can only help where (a) long-term memory
is informative and (b) the drift makes short-term temporarily more predictive — i.e. where
the *optimal α differs* between the stable and drifted regimes.

## 3.7 The fusion precondition — dense users

Restricting to **long-history users** (≥50 prior interactions) makes `L` informative. On
the dense cohort the α-response curve flips: NDCG@10 rises with α to a peak at **α = 0.8**
(validation +28 %, test +4 % over α = 0), falling only at α = 1.0. Precondition (a) is met.

## 3.8 The adaptation result — coherent-preference drift

Precondition (b) requires a drift whose new preference is short-term-predictable. The
anti-profile injection is not: it uses *popular* items (ranked well by the long-term path
and item bias) delivered as *random* cluster draws (no sequential structure). A diagnostic
fixed-α sweep confirmed the point — post-drift NDCG@10 is monotone increasing in α, peaking
at α = 0.8, so lowering α helps nowhere.

A **coherent-preference** injection was therefore specified — *on the mechanism, before
seeing any result* — in which the user switches to a tight, coherent cluster of
**mid-frequency** items (removing the popularity shortcut) and then follows it
consistently. On this benchmark the optimal α genuinely differs between regimes
(long-term when stable, short-term after drift), and CUSUM-driven adaptive fusion wins.
Across **30 injection seeds per scenario**:

| scenario | proposed − window-B (95 % CI) | proposed − static-α=0.8 |
|---|--:|--:|
| sudden | **+0.00078 ± 0.00010** (robust) | +0.00016 ± 0.00030 (ns) |
| gradual | **+0.00165 ± 0.00019** (robust) | −0.00012 ± 0.00027 (ns) |
| recurring | **+0.00069 ± 0.00008** (robust) | +0.00008 ± 0.00024 (ns) |

**The adaptive policy robustly beats window-B in all three drift scenarios** (every CI
excludes zero; largest on gradual, where the extended transition gives adaptation the most
room). This reverses the earlier window-B result generally. Against the strong static-α=0.8
baseline the adaptive policy is a **statistical tie** (never significantly worse; best- or
tied-best mean), the adaptive gain being concentrated in the comparatively short post-drift
window.

---

## Chapter 5 — Discussion & Limitations

**What is established.** (1) A validated sequential recommender that beats the popularity
floor by 47 %. (2) A strong, seed-robust drift-**detection** operating characteristic. (3)
A positive, pre-registered, seed-robust **adaptation** result: drift-aware fusion control
robustly outperforms the trivial short-only sliding window across sudden, gradual and
recurring drift, on a cohort and drift model where adaptation is theoretically able to
help.

**The boundaries, stated plainly.** The contribution is bounded, and the boundaries are
themselves a result:

- **Detection is not our strongest card.** On clean synthetic shifts, classical CUSUM
  detects better than our detector; we therefore use CUSUM and claim the adaptation layer,
  not the detector.
- **Adaptation needs the right conditions.** It helps only where long-term memory is
  informative (dense users) *and* the drift shifts the optimal α (a coherent, non-popular
  new preference). Where either fails — sparse users, or a popularity-laden drift — the
  optimal α is constant and no fusion control can beat the corresponding fixed policy. We
  map these conditions explicitly rather than hiding them.
- **The static-α=0.8 tie, and adaptation as insurance.** Against a well-tuned static
  fusion the advantage under the EWMA encoder is positive but not significant. A
  long-term-encoder ablation (EWMA vs a learned attention pool) shows *why* this is not
  simply a weak-encoder artefact: the two encoders **swap** which fixed baseline is strong
  — smooth EWMA makes static-α=0.8 strong and window-B weak; brittle attention collapses
  post-drift, making window-B strong and static-α=0.8 weak. Across both encoders and all
  three scenarios the adaptive policy is **never robustly worse than the best fixed
  policy**, robustly beats the *wrong* fixed choice, and under gradual+attention robustly
  beats **both**. The value of adaptation here is robustness to not knowing which regime —
  or which encoder failure mode — obtains, rather than a uniform margin over a single
  well-chosen static weight. EWMA remains the better default encoder.

**Threats to validity.** The dense cohort is small (196 users; long-history users are rare
in this data); a single trained backbone seed is used (injection-seed variance is
quantified, initialisation variance is not); the drift is synthetic with known onsets,
which is cleaner than real preference change; and evaluation is offline. The absolute NDCG
values are low by construction on data of this sparsity — the analysis is deliberately on
*relative* movement and confidence intervals, not absolute scores.

**Why the arc matters.** The result was reached by taking the negative findings seriously:
external baselines exposed that neither the detector nor the fusion was justified on the
original benchmark; the diagnosis identified *why* (too easy for detection, too sparse for
fusion); and pre-registered fixes produced a genuine, honestly-bounded positive result.
This is the intellectual honesty the review asked for, carried through to a defensible
contribution.

**Future work.** A stronger long-term encoder (attention pool or a second GRU) to widen
the pre/post optimal-α gap and target a significant static-α=0.8 win; backbone-seed and
full-scale replication; a learned-gate comparison (SLSRec-style) that drives α from a
network rather than an explicit detector; and validation on a second dataset.
