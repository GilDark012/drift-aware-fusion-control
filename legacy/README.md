# Legacy provenance scripts

These are the **original** data scripts that produced the frozen processed
parquet files under `data/processed/`. They are kept **verbatim** for provenance
and reproducibility record only.

| Script | Produced |
|---|---|
| `loader.py` | `data/processed/amazon_clean.parquet` (+ per-category intermediates) |
| `splitter.py` | `data/processed/{train,val,test}.parquet` + `split_cutoffs.json` |

**Status: frozen, not maintained.**

- They are **excluded from lint/type-check** (`legacy/` is in ruff's `extend-exclude`)
  because they are a historical record, not maintained package code.
- They reference `from src.utils.seed import set_all_seeds`, a module that no longer
  exists, and read raw JSONL from `F:/Dataset/raw` — so they are **not runnable as-is**.
  Do not run them; the data they produced is already frozen and validated
  (see `docs/phase2_data_validation.md`).
- The maintained, tested successors live in `src/drift_reco/data/` (schema, loading,
  validation). Per project scope, the ingestion/splitting **logic was intentionally not
  rewritten** — only re-homed here.
