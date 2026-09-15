# ADR-0002 — One shared backbone, α randomised during training

- **Statut** : accepté
- **Date** : 2026-08-18

## Contexte

Every baseline (A: fixed-α, B: sliding-window, C: periodic update, D: adaptive-α
without drift awareness) and the proposed drift-aware system must share **one**
recommender backbone and differ *only* in how the fusion weight α is chosen. This
is what isolates the contribution to α-control rather than to representation
quality (brief §13).

If the backbone is trained at a single fixed α (say 0.5), its representations are
optimised for that operating point; evaluating another policy at α=0.9 or an
adaptive α(t) then runs the backbone off-distribution and confounds the comparison.

## Options envisagées

| Option | Avantages | Inconvénients |
| --- | --- | --- |
| Train at fixed α=0.5 | simplest | backbone tuned to one α; unfair to other α policies; L or S path can atrophy |
| Train one backbone per α policy | each optimal | different representations → comparisons no longer isolate α-control; also breaks online adaptation (α changes without retraining) |
| **Train with α ~ U(0,1) per example** | both L and S stay independently predictive; any α policy uses the same fair backbone; matches online α-control | slightly higher variance; α=0/α=1 extremes seen only rarely |

## Décision

Train the shared backbone with **α sampled uniformly per example per step**
(`TrainConfig.alpha_train_mode='random'`). The long-term (`L`) and short-term (`S`)
paths are therefore both trained to be predictive on their own and in any mixture,
so downstream α policies — fixed baselines *and* the drift-aware controller — all
read from one identically-trained backbone. `alpha_train_mode='fixed'` is retained
as an ablation.

## Conséquences

- The recommender is trained **once**; baselines A–D and the proposed system are
  inference-time α policies over the same weights. Fair by construction.
- Enables the core mechanism: α is adapted online **without retraining**, which is
  the project's contribution (fusion control, not model retraining).
- Explicitly **excludes** claiming that any single α is "the" trained operating
  point; α is a control input, not a fixed hyperparameter of the backbone.
- Adds an ablation (fixed-α training) to quantify what α-randomisation costs/gains.
