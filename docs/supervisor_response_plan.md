# Supervisor Review — Response & Change Plan

Every point from the review mapped to a concrete action, its nature (code / experiment
/ writing), rough effort, whether it needs a decision, and status. Ordered by the
supervisor's own priorities. **Nothing in Chapter 3 should be re-written until the
backbone is fixed — the numbers will change.**

Legend: ☐ todo · ◐ in progress · ☑ done · ⟳ blocked-on-backbone · ⚑ needs your decision

---

## Priority ZERO — nothing else matters until these are done

### P0.1 — Backbone is broken (not just sparse) · experiment · ☑ **RESOLVED**
**Fixed by k-core filtering (ADR-0006).** Root cause was vocabulary sparsity (67%
singleton items), not the architecture. Model now beats the popularity floor
(NDCG 0.0171 vs 0.0116). Pipeline re-run: see `docs/rerun_fixed_backbone.md`. Detection
rose to **65% @ 2.4% false-alarm**. P0.2 floors ☑; P0.3 ☑ (short_term_drift kept, claim
rewritten — L–S ablation stays weak even on the working backbone). _Original note below._

### P0.1 (original) — diagnosis · ◐
**Decisive evidence (just computed):** the trained model is **below the popularity floor**.

| | Recall@10 | NDCG@10 | MRR |
|---|--:|--:|--:|
| Model (draft) | 0.00447 | 0.00224 | 0.00201 |
| **Popularity floor** | **0.00620** | **0.00270** | **0.00222** |
| Random floor | 4.5e-5 | 2.1e-5 | — |

She is right: a trained recommender that loses to "always recommend the 10 most popular
items" has learned nothing. **Two likely root causes, both in our training design:**
1. **α~U(0,1) random-fusion training** (ADR-0002) dilutes the gradient to each path.
2. **Uniform negative sampling** over a 220k vocab → the model wins by learning
   popularity alone (classic collapse).

**Action in progress:** retraining with **fixed α + popularity-based (∝ freq^0.75)
negatives + dim 128 + 15 epochs**. Decision point below.
- ⚑ **Decision:** if the retrain still does not clearly beat popularity, swap in a
  reference implementation (RecBole / official SASRec) rather than keep debugging ours.

### P0.2 — Add popularity & random floors · experiment · **done** · ☑
Computed above; will be added to Chapter 3 as the first table once the backbone is
re-run. Also report the **feasible-subset** figures (below).

### P0.3 — Detector must compare S to L, or rewrite the claim · **2 days** · ⟳ ⚑
The signal is `D=1−cos(S_u(t), EWMA_slow(S_u))` — self-consistency of the GRU state,
**not** divergence from `L_u(t)`. The abstract/Chapter-1 claim says the opposite.
- **Key insight:** we already have `ls_cosine` (the true L–S divergence) implemented as
  an ablation. It was unusable **because the backbone was broken** (L and S were
  near-orthogonal noise). **A working backbone may make L–S divergence meaningful** —
  so P0.1 may resolve this automatically.
- ⚑ **Decision (after retrain):** re-test `ls_cosine`; if usable → adopt it (matches the
  claim, L/S framing becomes real); if not → rewrite the abstract/Ch.1 to state the
  self-consistency signal plainly.

### P0.4 — Fully identify the dataset · writing · ☑ **done**
Data card rewritten with measured numbers: [docs/data_card.md](data_card.md). Source
**confirmed by the user**: **Amazon Reviews'23** (McAuley Lab, UCSD),
https://amazon-reviews-2023.github.io/ — cite **Hou et al. (2024), arXiv:2403.03952**
(*not* the 2018 Ni/Li/McAuley corpus I had first inferred). The full corpus spans
1996→2023; this project applies a **preprocessing time filter to 2014→2018-12-30**.
Categories {Electronics, Clothing_Shoes_and_Jewelry, Books} — **user confirmed** the three
raw category files were downloaded directly from the source page. 3,971,952 interactions,
130,987 users, 765,033 items, density 3.96×10⁻⁵. SHA-256 of processed artifacts + commit
recorded. **Licence resolved:** *no explicit licence* on either the source page or the
HuggingFace dataset page (verified 2026-08-27) — data card notes research-use + author
contact and that raw data is not redistributed. Only the exact **download date** remains
for the student to record.

### P0.5 — State the ranking protocol explicitly · writing · ☑ **done**
[docs/ranking_protocol.md](ranking_protocol.md): full-catalogue **unsampled** ranking
(cites Krichene & Rendle 2020), global chronological split, one positive/query ⇒
Recall@k=HitRate@k, causal context, thresholds tuned on validation only, evaluable
cohort ≥20 test interactions, seen-only + full reporting, popularity/random floors.

---

## Priority ONE

### P1.1 — Real baselines (sequential + drift) · experiment · ☑ **done (all three)**
- **Drift detection: Page-Hinkley / CUSUM / ADWIN** ☑ implemented dependency-free
  (`src/drift_reco/detectors/baselines.py`, unit-tested) and compared on the same signal
  (`scripts/baseline_detectors.py`, [docs/detector_baselines.md](detector_baselines.md)).
  **Finding:** on the clean synthetic shifts, **CUSUM detects 100 % @ 0 % false-alarm**
  and Page-Hinkley ~0.98 @ 0 %, both **strictly better than ours** (0.33 @ 0 %, needs
  ~11 % FA to reach 1.0); ADWIN fails (too sluggish on short streams). Honest conclusion:
  the adaptive z-score + persistence is **not justified on this benchmark** — it dilutes
  a strong signal. The regime where it should win (weak drift + heterogeneous volatility)
  is the real next experiment. DDM/EDDM cited but excluded (classifier-error detectors).
- ☑ Sequential recommenders **SASRec / GRU4Rec** (RecBole) — **done**. Installed RecBole
  1.2.1, exported the subsample to atomic files, ran both under the standard sequential
  protocol (`scripts/recbole_baseline.py`, [docs/sequential_baselines.md](sequential_baselines.md)).
  **SASRec 0.0201 / GRU4Rec 0.0151 / ours ~0.017 / floor 0.012** — same band, backbone
  validated (low absolute NDCG is the sparse data, not a bug). Protocol caveat noted
  (RecBole leave-one-out vs our global chronological cutoff).
- ☑ Long/short **SLSRec-style learned-gate** comparison — **done**. `src/drift_reco/adaptation/gate.py`
  + `scripts/learned_gate.py`, [docs/learned_gate_result.md](learned_gate_result.md).
  The explicit CUSUM-driven policy **robustly beats the learned gate in all three
  scenarios** (proposed−gate CIs exclude 0: sudden +0.00074, gradual +0.00161, recurring
  +0.00069). Folded into the report as Chapter 3.12. Co-trained / full-SLSRec gate is
  future work.

### P1.2 — Multiple seeds + confidence intervals · experiment · ☑ **done**
20-seed detection CIs (`seed_robustness.py`) **and** 30-seed × 3-scenario adaptation CIs
(`dense_adaptation_seeds.py`), with paired-difference significance (CIs excluding zero) on
every headline claim, plus the learned-gate and second-dataset comparisons. **Only
remaining axis:** backbone-initialisation seeds (each a 15-epoch retrain) — stated as a
limitation/future work, not a defect; the injection-seed variance (the main ask) is fully
quantified.

### P1.3 — Write the mechanism as equations · writing · ☑ **done**
[docs/method_equations.md](method_equations.md): full formal spec straight from code —
fusion `h=α·L+(1−α)·S` and scoring head, primary signal
`D=1−cos(S_t, S̄_t)` + the L–S ablation, causal z-score, the whole state machine, all
four α policies **including the gradual update rule**
`α_t=α_{t−1}+clip(α*(s_t)−α_{t−1}, −Δ, +Δ)`, and a **complete hyperparameter table**
(dim 128, W=10, h_L=20, h_D=30, z⁺=2.0/z⁻=0.8, p=r=3, warmup=8, Δ=0.05, lr=1e-3,
batch=512, epochs=15, n_neg=100, k-core=5 — the values actually used in the fixed run).

### P1.4 — Confront the window-B result · analysis · ☑ **done — reversed to a positive result**
Initially window-B *was* the ceiling on the sparse cohort ([window_b_analysis.md](window_b_analysis.md)):
`L` carried no signal, so no α>0 helped. Diagnosis led to two pre-registered fixes —
**dense long-history cohort** (so `L` is informative, [dense_cohort_probe.md](dense_cohort_probe.md))
and a **coherent-preference drift model** (so the optimal α differs pre/post,
[dense_adaptation_result.md](dense_adaptation_result.md)). On that principled benchmark
the adaptive policy **robustly beats window-B across sudden/gradual/recurring** (all 95 %
CIs exclude 0; 30 seeds), and an encoder ablation shows it is **never robustly worse than
the best fixed policy** across encoders/scenarios (adaptation-as-insurance). Ties the
strong static-α=0.8. The supervisor's sharpest criticism is answered with a
statistically-supported, honestly-bounded positive result.

### P1.5 — SLSRec citation + honest gap · writing · ☑ **done**
Metadata **verified from arXiv**: Zhou, Shen, Ji, Feng, Tang, He, Feng & Zhu (2026),
*SLSRec: Self-Supervised Contrastive Learning for Adaptive Fusion of Long- and Short-Term
User Interests*, **arXiv:2604.04530**. Cited in [bibliography.md](bibliography.md) (#14)
and folded into the report body (Chapter 1.5 + builder bibliography) with the gap stated:
we drive the fusion weight from an **explicit online drift detector**, not a learned
attention gate. Running the head-to-head comparison remains part of P1.1 (future work).

---

## Priority TWO — polish & completeness

- ☑ **P2.1** Title renamed "Adaptive Model **Switching**" → **"Adaptive Fusion Control"**
  in both report builders (there is no switching; α moves continuously).
- ☑ **P2.2** French **résumé accents restored** (via XML entities in the PDF builder;
  literal accents in the docx builder) — both résumés also updated to the current results.
- ☑ **P2.3** Every **"Still in progress"** removed from both builders (0 remaining);
  the student-fill boxes now read "To be completed by the student."
- ☑ **P2.4** Condition labels: Chapter 3 uses consistent descriptive names
  (static-α=0.8 / window-B / proposed-CUSUM); the skipped-C ambiguity is gone.
- ☑ **P2.5** Chapter 3 carries real tables throughout and 4 real figures; the front-matter
  Lists of Tables/Figures + the ToC are updated to match.
- ☑ **P2.6** General Introduction expanded; **dedicated Chapter 5 Discussion & Limitations
  added**; General Introduction, Ethics chapter and Appendices A–E all present.
- ☑ **P2.7** **Ethics / Data Protection chapter added** (Chapter 4 of the PDF builder):
  GDPR basis, consent/representativeness, and the false-alarm user-harm of an adaptive
  recommender.
- ☑ **P2.8** Category-change **reframed as a loose sanity indicator, not a drift
  definition** (phone-case-after-phone caveat); substantive evidence is the controlled
  injections, not category statistics.
- ☑ **P2.9** Bibliography expanded 7 → **~40 entries** and folded into the report builder
  (sequential, long/short, drift-detection, evaluation, dataset). ⚑ **Student must verify
  each citation against the source before submission** (academic integrity — I cannot fully
  verify all).
- ☑ **P2.10** Body now ~40+ pages of real content (47-page render), within the 40–80 range;
  further depth is optional and best added in the student's own voice.

---

## What she got right that we keep

Intellectual honesty (concede no general improvement), all 7 references verified, refusal
to over-claim on 11 users. **Preserve this tone** — where phrasing reads as evasion
("validating an auditable evaluation mechanism"), replace with the plain statement that
**the thesis currently has no positive ranking result**, which is defensible.

## Sequencing (matches her ordering)

1. **Backbone** (running) → re-run all Chapter-3 numbers → decide L-vs-S signal (P0.3).
2. Dataset ID + protocol + equations/hyperparameters (backbone-independent, do now).
3. Baselines + seeds/CIs.
4. Writing, ethics, figures, bibliography, French accents, title.
