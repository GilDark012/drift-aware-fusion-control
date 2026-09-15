# Run `20260826-131743__post-drift-alpha-probe`

- **Task**: post-drift-alpha-probe
- **Status**: failed
- **Started**: 2026-08-26T13:17:43.573514+00:00
- **Duration**: 9.51 s
- **Git commit**: 89fea675f0fc509ea79541285dab172bcddf12eb
- **Seed**: 42
- **Dataset**: amazon-subsample-dense

## Objective

Determine the post-drift-optimal fixed α to distinguish a long-term-favouring drift model from a mis-set adaptation target.

## Method

Sweep fixed α on the injected sudden streams; split pooled NDCG@10 into pre-onset and post-onset steps.

## Configuration

```json
{
  "alphas": [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
  ],
  "n_injected": 102
}
```

## Results

_No metric recorded._

## Findings

- DRIFT MODEL FAVOURS LONG-TERM: post-drift optimum is α=0.8 (0.01621) — dropping α cannot help; fix the injection.
- α=0.0: pre=0.01875 post=0.01163
- α=0.2: pre=0.01939 post=0.01304
- α=0.4: pre=0.01970 post=0.01415
- α=0.6: pre=0.01892 post=0.01531
- α=0.8: pre=0.02072 post=0.01621
- α=1.0: pre=0.01571 post=0.00000

## Limitations

- Dense cohort; single seed; sudden scenario.

## Next steps

- Fix the drift model, or accept and narrow the thesis.

## Error

```text
Traceback (most recent call last):
  File "C:\Users\gilal\Downloads\PFE_rec\scripts\post_drift_alpha_probe.py", line 125, in main
    _print(rows, verdict)
  File "C:\Users\gilal\Downloads\PFE_rec\scripts\post_drift_alpha_probe.py", line 151, in _print
    print(f"\nVERDICT: {verdict}")
  File "C:\Users\gilal\AppData\Local\Programs\Python\Python311\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u03b1' in position 64: character maps to <undefined>
```
