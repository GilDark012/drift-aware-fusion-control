# Explicit detector vs a learned fusion gate (P1.1, learned-gate half)

The project's central claim is that driving the long/short fusion weight from an
**explicit online drift detector** is preferable to an **implicit learned gate** of the
kind used by SLSRec and related long/short models. This experiment tests that claim
directly. `scripts/learned_gate.py` + `src/drift_reco/adaptation/gate.py`.

## Setup

A learned gate, `alpha = sigma(w . [L ; S] + b)`, is trained on top of the **frozen**
dense backbone to maximise next-item accuracy on the training stream (full-softmax
cross-entropy, in-vocabulary targets, 15 epochs). It is then exposed as an alpha-policy
and evaluated on the coherent-preference drift benchmark against the detector-driven
proposed policy and the static / window-B baselines. Both the gate and the proposed
policy choose alpha over the *same* frozen backbone, so they differ only in **how** the
fusion weight is decided — implicitly (a gate optimised for average training accuracy)
versus explicitly (an online CUSUM detector). Overall NDCG@10 across the drift is
reported per scenario over 20 injection seeds.

This is an SLSRec-*style* gate (a learned weighting over the same representations), not a
full re-implementation of SLSRec's contrastive pre-training; the comparison isolates the
gate-versus-detector control mechanism on identical representations.

## Result (20 seeds)

| scenario | proposed (detector) | learned gate | proposed - gate (95% CI) |
|---|--:|--:|--:|
| sudden | 0.01082 +/- 0.00041 | 0.01008 +/- 0.00038 | **+0.00074 +/- 0.00013** (robust) |
| gradual | 0.01534 +/- 0.00045 | 0.01373 +/- 0.00041 | **+0.00161 +/- 0.00025** (robust) |
| recurring | 0.00923 +/- 0.00023 | 0.00853 +/- 0.00025 | **+0.00069 +/- 0.00009** (robust) |

## Reading

**The explicit detector-driven policy robustly beats the learned gate in every scenario**
(each paired-difference confidence interval excludes zero), with the largest margin on
gradual drift. The learned gate is a reasonable fusion - it sits between window-B and the
proposed policy - but it is optimised for *average* accuracy over the training
distribution and therefore has no explicit notion of *when* an individual user's
behaviour has shifted. On a stream containing a genuine, coherent preference change, an
online detector that reacts to that change at the moment it is confirmed does better than
a static learned weighting, and it does so while remaining interpretable and tunable to
an operating point.

This is the empirical form of the gap stated in Chapter 1: the contribution is not a new
representation or a better average-case fusion, but the demonstration that an **explicit,
online, interpretable** drift decision can control the fusion at least as well as - here,
better than - an implicit learned gate, on drift that the gate's training objective does
not specifically target.

## Limitations

Single backbone seed and the small dense cohort, as elsewhere. The gate is a linear (or
one-hidden-layer) fusion over the frozen representations; a gate co-trained with the
backbone, or SLSRec's full contrastive objective, could narrow or change the gap and is
left as future work. The comparison is on the coherent-drift benchmark, where the optimal
alpha differs between regimes; on a stream with no such shift a learned gate and a static
policy would be expected to coincide.
