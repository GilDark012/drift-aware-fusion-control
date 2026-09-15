# Phase 1 — Repository Audit

**Project:** Drift-Aware Long/Short-Term Fusion for Sequential Recommenders (PFE)
**Author:** Gil-Allen Mounzeo · Aivancity (PGE5)
**Audit date:** 2026-08-17
**Auditor:** ML engineering pass (read-only; no code modified)

---

## 1. What physically exists in the repository

```
PFE_rec/
├── Drift_rec_sys_Scope_Note.md      # Phase-1 scope note (June 2026)
├── experiment_protocol.docx         # protocol (not yet read)
├── data_example.txt                 # data card (partly a template — has [placeholders])
├── loader.py                        # Amazon ingestion  (root, NOT under src/)
├── splitter.py                      # temporal 70/10/20 split (root, NOT under src/)
└── data/
    ├── raw/                         # empty (raw jsonl lives on F:/Dataset/raw)
    ├── external/                    # empty
    ├── processed/
    │   ├── amazon_clean.parquet             # 3,971,952 rows  ✅
    │   ├── train.parquet                    # 2,780,366 rows (70.0%) ✅
    │   ├── val.parquet                      #   397,195 rows (10.0%) ✅
    │   ├── test.parquet                     #   794,391 rows (20.0%) ✅
    │   ├── {Electronics,Clothing,Books}_filtered.parquet   # per-category intermediates
    │   └── {cat}_parts/                     # streamed raw batches (hundreds of files)
    └── drift/
        ├── test_control.parquet             # 794,391 rows (= test, un-drifted control) ✅
        ├── test_drifted.parquet             # 794,391 rows (test with injected drift) ✅
        └── drift_ground_truth.json          # D1 sudden / D2 gradual / D3 recurring onsets ✅
```

### Verified data schema (all splits identical)
`user_id:str, item_id:str, category:str, preference_score:double, timestamp:ns, user_idx:int64, item_idx:int64`

### Verified counts
| Split | Rows | Share |
|---|---|---|
| amazon_clean | 3,971,952 | 100% |
| train | 2,780,366 | 70.00% |
| val | 397,195 | 10.00% |
| test | 794,391 | 20.00% |

The temporal 70/10/20 split is **exact** and consistent with `splitter.py`.

### Controlled-drift artifacts (already generated)
`drift_ground_truth.json` gives three onset points **as row indices into the test stream**:
- **D1 sudden** — idx 158,878 · 2018-02-14
- **D2 gradual** — idx 397,195 · 2018-05-25
- **D3 recurring** — idx 595,793 · 2018-08-30

`test_control.parquet` (no drift) and `test_drifted.parquet` (drift injected) are paired, same length — this is the ground-truth harness for measuring detection/recovery latency.

---

## 2. What is implemented vs. missing

| Component | Status | Notes |
|---|---|---|
| Amazon ingestion (`loader.py`) | ✅ works, outputs present | but `import from src.utils.seed` — **src/ does not exist** |
| Temporal split (`splitter.py`) | ✅ works, outputs present | same broken `src.utils.seed` import |
| Processed parquet (clean + splits) | ✅ present & valid | reusable as-is |
| Controlled drift injection | ✅ artifacts present | **injection *code* not in repo** — only outputs |
| `src/` package (models/drift/fusion/eval) | ❌ absent | nothing exists |
| Long/short-term recommender | ❌ absent | to build (Phase 3) |
| Divergence / drift signal | ❌ absent | to build (Phase 4) |
| Drift detector (persistence-aware) | ❌ absent | to build (Phase 5) |
| Adaptive α fusion | ❌ absent | to build (Phase 6) |
| Evaluation / latency framework | ❌ absent | to build (Phase 8) |
| Baselines & ablations | ❌ absent | to build (Phase 8) |
| configs / tests / notebooks / results | ❌ absent | to build |
| Git repository | ❌ not initialized | — |
| pyproject / lab-journal infra | ❌ absent | scaffolding needed |

**Bottom line:** the *data layer* is done and reusable. Essentially **all research code is still to be written**. The two root scripts are the only Python, and they have a dangling `src.utils.seed` import (the outputs were clearly produced under an earlier layout that no longer exists).

---

## 3. Risks & things to reconcile

1. **Framing conflict (most important).** The existing `Drift_rec_sys_Scope_Note.md` frames the project as a **generic concept-drift detection framework** — SLi-Rec base, PUDD + KL Embedding-Divergence + Performance-Degradation detectors, ensemble voting, DTD dynamic threshold, AdaMoE baseline, Docker, `<100ms` latency, `<2GB` memory. The **new project brief explicitly reframes** the contribution as *drift-aware control of long/short-term fusion* and lists "turn this into a generic concept-drift framework" as **failure mode #1**. These two documents pull in different directions. This must be reconciled before building (see decision below).

2. **"Injection code" is missing.** We have drifted test data but not the script that produced it. For the brief's requirement *"the injected shift must affect the preference sequence, not just flip a category label"* and *"ground-truth onset must be known,"* we need to re-derive or re-implement the injection so it is reproducible and defensible. The existing parquet can be kept as a frozen artifact, but the generating code should exist.

3. **Global-split semantics.** `splitter.py` cuts on a **global timestamp**, so a single user's interactions are distributed across train/val/test by time. This is *correct and necessary* for per-user temporal drift evaluation, but the script's comment ("NO user interaction may appear in more than one split") conflates *row* disjointness (true) with *user* disjointness (false, and intentionally so). No leakage exists; the comment is just misleading. Worth noting for the report.

4. **Cold-start in test.** With a global split, some test-period users may have little/no train history → their long-term embedding is weak. Needs quantifying in Phase 2 (data validation) because it affects per-user latency metrics.

5. **Scale.** 794k test interactions streamed row-by-row through a neural recommender + per-step drift loop is heavy. A development subsample (stratified by user, drift-onset-preserving) will be needed for fast iteration, with full-scale runs reserved for final numbers.

6. **Broken imports.** `loader.py` / `splitter.py` reference `src.utils.seed.set_all_seeds`, which does not exist. Low effort to restore; needed only if we re-run ingestion (we should not need to — outputs exist).

---

## 4. Recommended implementation plan (aligned to the brief's phases)

- **Phase 1 (this doc)** — audit ✅
- **Phase 2** — data validation report (counts, category mix, per-user activity, cold-start rate, drift-onset sanity, leakage checks) on the *existing* parquet; no re-ingestion.
- **Scaffold** — initialize the lab-journal repo structure (`src/drift_reco/…`, configs, artifacts, JOURNAL) so every run is traceable.
- **Phase 3** — long/short-term recommender with explicit `L_u(t)`, `S_u(t)`, fixed-α fusion baseline.
- **Phase 4** — divergence signal `D_u(t)` + diagnostic plots on representative users.
- **Phase 5** — persistence-aware detector; fixed vs dynamic thresholds (tuned on val only).
- **Phase 6** — adaptive α (gradual vs abrupt).
- **Phase 7** — reproducible sudden/gradual/recurring injection *code* (re-derive onsets), keep existing parquet as frozen reference.
- **Phase 8** — full eval: baselines A–D + proposed + ablations, latency triplet (detect/adapt/recover).
- **Phase 9** — analysis incl. user heterogeneity.
- **Phase 10** — docs, tables, figures, limitations.

**Reuse, do not rebuild:** `amazon_clean.parquet`, `train/val/test.parquet`, the drift ground-truth + control/drifted parquet. Do **not** re-run `loader.py`/`splitter.py` unless a data defect is proven.
