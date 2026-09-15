#!/usr/bin/env python3
"""P1.1 (learned-gate half) — explicit detector vs a learned fusion gate.

Trains an SLSRec-style learned fusion gate on top of the frozen dense backbone and
compares it, on the coherent-preference drift benchmark, against the detector-driven
proposed policy and the static / window-B baselines. Both the gate and the proposed
policy choose alpha over the *same* frozen backbone, so they differ only in how the
fusion weight is decided: implicitly (a learned gate optimised for training accuracy)
versus explicitly (an online CUSUM detector). Overall NDCG@10 across the drift is
reported per scenario with 95% confidence intervals over injection seeds.

Assumes the dense subsample and backbone exist (run scripts/dense_adaptation.py first).

Usage:
    python scripts/learned_gate.py --seeds 20 --gate-epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dense_adaptation import (  # noqa: E402
    DENSE_DIR,
    FA_TARGET,
    STABLE_ALPHA,
    TOP_K,
    _calibrate,
    _ckpt,
    _prepare,
    _region_metrics,
    _run_policy,
)
from dense_adaptation_seeds import _ci95  # noqa: E402

from drift_reco.adaptation.gate import (  # noqa: E402
    GatePolicy,
    GateTrainConfig,
    train_gate,
)
from drift_reco.adaptation.policy import AdaptationConfig, build_policy  # noqa: E402
from drift_reco.config.constants import S_USER_COLUMN  # noqa: E402
from drift_reco.data.injection import DriftInjector, InjectionConfig  # noqa: E402
from drift_reco.data.schema import Split  # noqa: E402
from drift_reco.data.subsample import read_subsample  # noqa: E402
from drift_reco.detectors.baselines import (  # noqa: E402
    CusumDetectorConfig,
    CusumDriftDetector,
)
from drift_reco.detectors.signal import get_signal  # noqa: E402
from drift_reco.evaluation.streaming import StreamingEvaluator  # noqa: E402
from drift_reco.models.checkpoint import load_backbone  # noqa: E402
from drift_reco.models.sequences import user_sequences  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

SCENARIOS = ("sudden", "gradual", "recurring")


def _overall(evaluator, detector, ctx_by, inj_by, policy, onset_by_user) -> float:
    """Overall NDCG@10 of one policy object over the injected users."""
    results = [
        r
        for user, frame in inj_by.items()
        if user in ctx_by
        for r in [evaluator.run_user(ctx_by[user], frame, detector, policy)]
        if r is not None
    ]
    return _region_metrics(results, onset_by_user)["all"].ndcg_at_k


def _policies(gate_policy: GatePolicy) -> dict:
    """The policy objects to compare (three baselines built by config + the gate)."""
    return {
        "static-0.8": build_policy(
            AdaptationConfig(policy="static", alpha_static=STABLE_ALPHA)
        ),
        "window-B": build_policy(AdaptationConfig(policy="static", alpha_static=0.0)),
        "proposed-CUSUM": build_policy(AdaptationConfig(
            policy="state", alpha_stable=STABLE_ALPHA, alpha_emerging=STABLE_ALPHA,
            alpha_confirmed=0.2, mode="gradual", alpha_step=0.05,
        )),
        "learned-gate": gate_policy,
    }


def main() -> int:
    """Train the learned gate and compare it to the detector-driven policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--gate-epochs", type=int, default=10)
    parser.add_argument("--gate-hidden", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, _ = read_subsample(DENSE_DIR)
    model, meta = load_backbone(_ckpt("ewma"), device)
    evaluator = StreamingEvaluator(
        model, meta.architecture, get_signal("short_term_drift"), device, top_k=TOP_K
    )
    ctx_by, control_by, lookups, anchors = _prepare(
        model, meta.architecture, frames, device, "coherent"
    )
    gate = train_gate(
        model, meta.architecture, user_sequences(frames[Split.TRAIN]), device,
        GateTrainConfig(
            epochs=args.gate_epochs, hidden=args.gate_hidden, seed=args.seed
        ),
    )
    gate_policy = GatePolicy(gate)

    control = _run_policy(
        evaluator, CusumDriftDetector(), ctx_by, control_by,
        AdaptationConfig(policy="static", alpha_static=STABLE_ALPHA),
    )
    enter = _calibrate(control, FA_TARGET)
    detector = CusumDriftDetector(CusumDetectorConfig(enter=enter))
    policies = _policies(gate_policy)

    config = RunConfig(
        task="p1-1-learned-gate",
        description="Explicit detector vs learned fusion gate (coherent drift)",
        params={"seeds": args.seeds, "gate_epochs": args.gate_epochs,
                "gate_hidden": args.gate_hidden, "scenarios": list(SCENARIOS)},
        dataset="amazon-subsample-dense",
        tags=["p1-1", "learned-gate", "coherent"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        rows = _sweep(evaluator, detector, ctx_by, lookups, anchors,
                      frames[Split.TEST], policies, args)
        journal.table(rows, "learned_gate_comparison")
        _summarize(journal, rows, enter)
        _print(rows)
    return 0


def _sweep(evaluator, detector, ctx_by, lookups, anchors, test, policies, args) -> list:
    """Per-scenario mean +/- CI for every policy, plus the proposed-vs-gate diff."""
    rows = []
    for scenario in SCENARIOS:
        per_policy = {name: [] for name in policies}
        diff = []
        for seed in range(args.seed, args.seed + args.seeds):
            injected, onsets = DriftInjector(
                InjectionConfig(drift_type=scenario, mechanism="coherent",
                                n_users=5000, seed=seed, min_test_len=8),
                lookups,
            ).inject(test, anchors)
            inj_by = dict(iter(injected.groupby(S_USER_COLUMN)))
            onset_by_user = {int(o.user): int(o.onset_index) for o in onsets}
            vals = {
                name: _overall(evaluator, detector, ctx_by, inj_by, pol, onset_by_user)
                for name, pol in policies.items()
            }
            for name, v in vals.items():
                per_policy[name].append(v)
            diff.append(vals["proposed-CUSUM"] - vals["learned-gate"])
        row = {"scenario": scenario}
        for name, vs in per_policy.items():
            mean, half = _ci95(vs)
            row[name] = f"{mean:.5f}+/-{half:.5f}"
        dmean, dhalf = _ci95(diff)
        row["proposed_minus_gate"] = f"{dmean:+.5f}+/-{dhalf:.5f}"
        row["proposed_beats_gate_robust"] = bool(dmean - dhalf > 0)
        rows.append(row)
    return rows


def _summarize(journal: LabJournal, rows: list, enter: float) -> None:
    """Write the run's narrative results."""
    journal.summary(
        objective="Compare an explicit CUSUM-driven fusion policy against a learned "
        "fusion gate (SLSRec-style) on the coherent-drift benchmark.",
        method=f"Train the gate on the frozen dense backbone; CUSUM enter={enter:g}; "
        "compare overall NDCG@10 per scenario with 95% CIs over injection seeds.",
        findings=[
            f"[{r['scenario']}] proposed {r['proposed-CUSUM']} vs learned-gate "
            f"{r['learned-gate']}; diff {r['proposed_minus_gate']} "
            f"({'robust' if r['proposed_beats_gate_robust'] else 'ns'})"
            for r in rows
        ],
        limitations=["Dense cohort; one backbone seed; gate is a linear/MLP fusion."],
        next_steps=["Second-dataset validation."],
    )


def _print(rows: list) -> None:
    """Echo the comparison to stdout."""
    print("\nExplicit detector policy vs learned gate (overall NDCG@10):")
    for r in rows:
        tag = "robust" if r["proposed_beats_gate_robust"] else "ns"
        print(f"  [{r['scenario']:<9}] proposed {r['proposed-CUSUM']} | "
              f"gate {r['learned-gate']} | diff {r['proposed_minus_gate']} {tag}")


if __name__ == "__main__":
    raise SystemExit(main())
