# Drift-Aware Fusion Control

> Drift-aware long/short-term recommendation with adaptive fusion control for
> user preference evolution. Research code for a PFE thesis (aivancity PGE5,
> 2025–2026), evaluated on the public Amazon Reviews'23 corpus.

## Research question

How can drift awareness help a long/short-term fusion recommender recognise
that the current balance between stable and recent preferences has become
inappropriate, and adapt that balance faster while remaining robust to
temporary behavioural noise?

## Method overview

| Component | Role |
| --- | --- |
| Frozen backbone | GRU short-term encoder + recency-weighted long-term pool, trained once; every policy is compared on identical representations |
| Drift signal | Causal divergence between the short-term state and its own slow EWMA reference |
| Detector | Per-user standardised score driving a persistence-aware state machine (STABLE → EMERGING → CONFIRMED, with a TEMPORARY branch); an off-the-shelf CUSUM is adopted downstream after it won the detector comparison |
| Fusion policy | The fusion weight α glides toward a state-dependent target, rate-limited to Δ = 0.05 per step — control, never model switching |
| Evaluation | Full-catalogue ranking, global chronological split, controlled drift injection with known onsets, paired confidence intervals over injection seeds |

## Key findings

- The backbone beats the popularity floor by 47% (NDCG@10 0.0171 vs 0.0116)
  and is corroborated by external SASRec and GRU4Rec baselines.
- On a dense-user cohort under a pre-registered coherent-preference drift
  model, drift-aware fusion control robustly beats a short-only sliding window
  and a learned fusion gate across sudden, gradual and recurring drift; it is
  statistically tied with a well-tuned static fusion.
- An encoder ablation and a second-domain replication show the durable
  property is robustness: the adaptive policy is never the worst and beats
  whichever fixed choice is wrong for the setting.
- Negative results are reported in full; they shaped the investigation.

## Scope

The prototype is a proof of the methodology, not a production system: the
study is offline, on public data, with no live traffic. The controller
requires three hooks in the host recommender — exposed user representations,
an adjustable fusion weight α, and an online feedback signal.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Usage

```bash
make help          # list the available targets
make check         # full compliance gate (structure, style, types, tests)
make test          # unit-test suite (121 tests)
```

Reproduction entry points for each stage of the pipeline:

```bash
python scripts/build_subsample.py --k-core 5        # development subsample
python scripts/train_baseline.py --epochs 15        # train and evaluate the backbone
python scripts/analyze_divergence.py                # drift-signal analysis
python scripts/detect_drift.py                      # detector calibration
python scripts/inject_drift.py                      # controlled drift injection
python scripts/full_evaluation.py                   # detect-adapt-recover evaluation
python scripts/seed_robustness.py --seeds 20        # detection seed robustness
python scripts/baseline_detectors.py                # CUSUM / Page-Hinkley / ADWIN comparison
python scripts/dense_adaptation_seeds.py --seeds 30 # dense-cohort adaptation with CIs
python scripts/learned_gate.py --seeds 20           # learned-gate comparison
python scripts/second_dataset.py --category Electronics  # second-domain replication
python scripts/recbole_baseline.py --epochs 30      # external SASRec / GRU4Rec
```

## Repository structure

```
docs/         scoping, architecture, experimental protocol, data card, results, ADRs
src/          source code (single source of truth for the logic)
tests/        unit tests, mirroring src/
scripts/      command-line entry points
notebooks/    exploration only
data/         local data, not versioned (see data/README.md)
artifacts/    one timestamped folder per run + an index.csv registry
reports/      generated compliance reports
JOURNAL.md    chronological run journal
```

## Traceability

Every experiment writes an `artifacts/runs/<RUN_ID>/` folder containing the
validated configuration, the metrics (`metrics.csv`), the event trace
(`events.csv`), the full log (`run.log`), a readable summary (`results.md`),
free-form notes and a provenance manifest (Git commit, versions, file
fingerprints). No published number exists without the run that produced it.

## Results

Consolidated results live in `docs/` (see `docs/results.md` and the phase
result documents). Absolute NDCG@10 values are low: that is the expected
behaviour on very sparse Amazon data, and the analysis rests on relative,
seed-robust differences against pre-drift baselines.

## Data

The raw Amazon Reviews'23 files (Hou et al., 2024,
https://amazon-reviews-2023.github.io/) are not redistributed here; download
them from the source and see `docs/data_card.md` for provenance, filtering and
checksums.

## Licence

See `LICENSE`.
