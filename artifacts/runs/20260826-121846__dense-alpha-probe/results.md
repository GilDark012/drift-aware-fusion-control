# Run `20260826-121846__dense-alpha-probe`

- **Task**: dense-alpha-probe
- **Status**: failed
- **Started**: 2026-08-26T12:18:46.295272+00:00
- **Duration**: 66.68 s
- **Git commit**: 41905e512b6bdcc7bb86fed876139f1ddaae11c3
- **Seed**: 42
- **Dataset**: amazon-subsample-dense

## Objective

Test the precondition for the adaptation thesis: on a dense long-history cohort, does any α>0 (using long-term memory) beat α=0?

## Method

Build a dense subsample (train-history floor); train with random-α so both paths are optimised; sweep α at inference on val and test.

## Configuration

```json
{
  "dense": {
    "k_core": 5,
    "min_test": 5,
    "min_train": 50
  },
  "n_items": 11581,
  "alphas": [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
  ]
}
```

## Results

| metric | split | step | value | unit | context |
| --- | --- | --- | --- | --- | --- |
| train_loss | train | 0 | 3.2048154904110597 |  |  |
| ndcg_at_10 | test | 0 | 0.01624 |  |  |
| ndcg_at_10 | test | 20 | 0.01666 |  |  |
| ndcg_at_10 | test | 40 | 0.01688 |  |  |
| ndcg_at_10 | test | 60 | 0.01669 |  |  |
| ndcg_at_10 | test | 80 | 0.0169 |  |  |
| ndcg_at_10 | test | 100 | 0.01361 |  |  |

## Findings

- L HELPS: best α=0.8 beats α=0 by +4.1% -> adaptation has room.
- test NDCG@10: α=0 -> 0.01624; best α=0.8 -> 0.01690
- val sweep: α0.0=0.01834, α0.2=0.01960, α0.4=0.01946, α0.6=0.02153, α0.8=0.02346, α1.0=0.01985
- test sweep: α0.0=0.01624, α0.2=0.01666, α0.4=0.01688, α0.6=0.01669, α0.8=0.01690, α1.0=0.01361

## Limitations

- Dense cohort is small (long-history users are rare); one trained seed.
- EWMA-pool long-term encoder; a stronger L encoder is a separate lever.

## Next steps

- If L helps: re-run injection + adaptation (CUSUM) on this cohort.
- If not: report the negative result and narrow the thesis to detection.

## Error

```text
Traceback (most recent call last):
  File "C:\Users\gilal\Downloads\PFE_rec\scripts\dense_alpha_probe.py", line 179, in main
    _print(val_rows, test_rows, verdict)
  File "C:\Users\gilal\Downloads\PFE_rec\scripts\dense_alpha_probe.py", line 221, in _print
    print("\nα-sweep NDCG@10 (higher = better):")
  File "C:\Users\gilal\AppData\Local\Programs\Python\Python311\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u03b1' in position 2: character maps to <undefined>
```
