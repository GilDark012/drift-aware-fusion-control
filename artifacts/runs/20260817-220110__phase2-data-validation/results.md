# Run `20260817-220110__phase2-data-validation`

- **Task**: phase2-data-validation
- **Status**: failed
- **Started**: 2026-08-17T22:01:10.829460+00:00
- **Duration**: 8.38 s
- **Git commit**: n/a
- **Seed**: 42
- **Dataset**: amazon-processed

## Objective

Certify processed Amazon splits: counts, leakage, drift ground truth

## Method

See `config.json` for the exact parameters.

## Configuration

```json
{
  "processed_dir": "data/processed",
  "drift_dir": "data/drift"
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| n_rows | train | 0 | 2780366.0 | rows |  |
| n_users | train | 0 | 127344.0 | users |  |
| n_items | train | 0 | 589045.0 | items |  |
| median_interactions_per_user | train | 0 | 20.0 |  |  |
| n_rows | val | 0 | 397195.0 | rows |  |
| n_users | val | 0 | 81998.0 | users |  |
| n_items | val | 0 | 183070.0 | items |  |
| median_interactions_per_user | val | 0 | 3.0 |  |  |
| n_rows | test | 0 | 794391.0 | rows |  |
| n_users | test | 0 | 98852.0 | users |  |
| n_items | test | 0 | 286622.0 | items |  |
| median_interactions_per_user | test | 0 | 6.0 |  |  |
| test_cold_start_user_rate | test | 0 | 0.034627523975235705 |  |  |
| test_new_item_rate | test | 0 | 0.48605480388804767 |  |  |
| duplicate_rows_across_splits | all | 0 | 0.0 | rows |  |

## Figures

![user-activity-distribution.png](figures/user-activity-distribution.png)

## Notes

```text
[2026-08-17T22:01:18.432800+00:00] WARNING: pre-generated test_drifted.parquet is index-inconsistent (item_id changed but item_idx unchanged) — not usable for training; controlled drift must be re-injected at the preference-sequence level
```

## Error

```text
Traceback (most recent call last):
  File "C:\Users\gilal\Downloads\PFE_rec\scripts\validate_data.py", line 94, in main
    _save_figures(journal, validator)
  File "C:\Users\gilal\Downloads\PFE_rec\scripts\validate_data.py", line 68, in _save_figures
    journal.figure(figures.category_share_over_time(), "category_share_over_time")
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\Downloads\PFE_rec\src\drift_reco\data\plots.py", line 61, in category_share_over_time
    full["period"] = full[TIMESTAMP_COLUMN].dt.to_period(freq).dt.to_timestamp()
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\core\accessor.py", line 112, in f
    return self._delegate_method(name, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\core\indexes\accessors.py", line 132, in _delegate_method
    result = method(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\core\indexes\extension.py", line 95, in method
    result = attr(self._data, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\core\arrays\datetimes.py", line 1249, in to_period
    return PeriodArray._from_datetime64(self._ndarray, freq, tz=self.tz)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\core\arrays\period.py", line 331, in _from_datetime64
    data, freq = dt64arr_to_periodarr(data, freq, tz)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gilal\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\core\arrays\period.py", line 1193, in dt64arr_to_periodarr
    freq = Period._maybe_convert_freq(freq)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "period.pyx", line 1765, in pandas._libs.tslibs.period._Period._maybe_convert_freq
  File "offsets.pyx", line 4965, in pandas._libs.tslibs.offsets.to_offset
ValueError: MS is not supported as period frequency
```
