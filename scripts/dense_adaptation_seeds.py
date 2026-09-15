#!/usr/bin/env python3
"""Seed-robustness of the CUSUM adaptive-fusion win (coherent-preference drift).

The single-seed run showed proposed-CUSUM beating both static-α=0.8 and window-B on the
coherent-preference drift. The margin is modest, so before claiming a win we repeat the
comparison over many injection seeds and report the paired difference
(proposed − static-0.8) with a 95 % confidence interval and the fraction of seeds the
adaptive policy wins.

Reuses the dense-adaptation machinery; assumes the dense subsample and backbone exist
(run `scripts/dense_adaptation.py` first).

Usage:
    python scripts/dense_adaptation_seeds.py --seeds 12
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
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

from drift_reco.adaptation.policy import AdaptationConfig  # noqa: E402
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
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402

POLICIES = {
    "static-0.8": AdaptationConfig(policy="static", alpha_static=STABLE_ALPHA),
    "window-B": AdaptationConfig(policy="static", alpha_static=0.0),
    "proposed-CUSUM": AdaptationConfig(
        policy="state", alpha_stable=STABLE_ALPHA, alpha_emerging=STABLE_ALPHA,
        alpha_confirmed=0.2, mode="gradual", alpha_step=0.05,
    ),
}


def _ci95(values: list[float]) -> tuple[float, float]:
    """Return (mean, 95 % half-width) of a sample."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return float(arr.mean()) if arr.size else float("nan"), float("nan")
    return float(arr.mean()), 1.96 * float(arr.std(ddof=1)) / math.sqrt(arr.size)


def _one_seed(evaluator, detector, ctx_by, inj_by, onset_by_user) -> dict[str, float]:
    """Overall NDCG@10 per policy for one injected realisation."""
    out = {}
    for name, cfg in POLICIES.items():
        results = _run_policy(evaluator, detector, ctx_by, inj_by, cfg)
        out[name] = _region_metrics(results, onset_by_user)["all"].ndcg_at_k
    return out


def _aggregate(per_seed: list[dict[str, float]]) -> dict:
    """Mean ± CI per policy, paired proposed−static diff, and win rate."""
    names = list(POLICIES)
    means = {n: _ci95([s[n] for s in per_seed]) for n in names}
    diff = [s["proposed-CUSUM"] - s["static-0.8"] for s in per_seed]
    diff_vs_b = [s["proposed-CUSUM"] - s["window-B"] for s in per_seed]
    wins = sum(
        1 for s in per_seed
        if s["proposed-CUSUM"] > s["static-0.8"] and s["proposed-CUSUM"] > s["window-B"]
    )
    return {
        "means": means,
        "diff_static": _ci95(diff),
        "diff_window": _ci95(diff_vs_b),
        "win_rate": wins / len(per_seed),
        "n_seeds": len(per_seed),
    }


def _run_scenario(
    evaluator, detector, ctx_by, lookups, anchors, test, scenario, seeds, base_seed
) -> dict:
    """Seed loop for one drift scenario; return the aggregated comparison."""
    per_seed = []
    for seed in range(base_seed, base_seed + seeds):
        injected, onsets = DriftInjector(
            InjectionConfig(drift_type=scenario, mechanism="coherent",
                            n_users=5000, seed=seed, min_test_len=8),
            lookups,
        ).inject(test, anchors)
        inj_by = dict(iter(injected.groupby(S_USER_COLUMN)))
        onset_by_user = {int(o.user): int(o.onset_index) for o in onsets}
        per_seed.append(_one_seed(evaluator, detector, ctx_by, inj_by, onset_by_user))
    return _aggregate(per_seed)


def main() -> int:
    """Repeat the adaptive-vs-baselines comparison over seeds, per scenario."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--scenarios", default="sudden,gradual,recurring")
    parser.add_argument(
        "--long-term-mode", default="ewma", choices=["mean", "ewma", "attention"]
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, _ = read_subsample(DENSE_DIR)
    model, meta = load_backbone(_ckpt(args.long_term_mode), device)
    evaluator = StreamingEvaluator(
        model, meta.architecture, get_signal("short_term_drift"), device, top_k=TOP_K
    )
    ctx_by, control_by, lookups, anchors = _prepare(
        model, meta.architecture, frames, device, "coherent"
    )

    config = RunConfig(
        task="dense-adaptation-seeds",
        description="Seed-robustness of CUSUM adaptive fusion, per scenario (coherent)",
        params={"seeds": args.seeds, "mechanism": "coherent", "scenarios": scenarios},
        dataset="amazon-subsample-dense",
        tags=["adaptation", "seeds", "coherent"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        # Calibrate CUSUM once on the (seed- and scenario-invariant) control streams.
        control = _run_policy(
            evaluator, CusumDriftDetector(), ctx_by, control_by,
            AdaptationConfig(policy="static", alpha_static=STABLE_ALPHA),
        )
        enter = _calibrate(control, FA_TARGET)
        detector = CusumDriftDetector(CusumDetectorConfig(enter=enter))

        by_scenario = {
            scenario: _run_scenario(
                evaluator, detector, ctx_by, lookups, anchors,
                frames[Split.TEST], scenario, args.seeds, args.seed,
            )
            for scenario in scenarios
        }
        _report(journal, by_scenario, enter)
    return 0


def _scenario_rows(by_scenario: dict) -> list[dict]:
    """One row per (scenario, policy): mean overall NDCG@10 and its CI."""
    rows = []
    for scenario, agg in by_scenario.items():
        for name, (mean, half) in agg["means"].items():
            rows.append({
                "scenario": scenario, "policy": name,
                "overall_ndcg_mean": round(mean, 5), "ci95": round(half, 5),
            })
    return rows


def _diff_rows(by_scenario: dict) -> list[dict]:
    """Per-scenario paired differences (proposed - each baseline) with robustness."""
    rows = []
    for scenario, agg in by_scenario.items():
        dm, dh = agg["diff_static"]
        wm, wh = agg["diff_window"]
        rows.append({
            "scenario": scenario,
            "vs_window_B": f"{wm:+.5f}+/-{wh:.5f}",
            "vs_window_B_robust": bool(wm - wh > 0),
            "vs_static_08": f"{dm:+.5f}+/-{dh:.5f}",
            "vs_static_08_robust": bool(dm - dh > 0),
            "win_rate": round(agg["win_rate"], 3),
            "n_seeds": agg["n_seeds"],
        })
    return rows


def _report(journal: LabJournal, by_scenario: dict, enter: float) -> None:
    """Log and print the per-scenario robustness comparison."""
    means_rows = _scenario_rows(by_scenario)
    diff_rows = _diff_rows(by_scenario)
    journal.table(means_rows, "adaptation_seed_robustness")
    journal.table(diff_rows, "adaptation_paired_diffs")
    for d in diff_rows:
        journal.metric("proposed_minus_window_robust", float(d["vs_window_B_robust"]),
                       context=d["scenario"])
        journal.metric("proposed_minus_static_robust", float(d["vs_static_08_robust"]),
                       context=d["scenario"])
    findings = [
        f"[{d['scenario']}] vs window-B {d['vs_window_B']} "
        f"({'ROBUST' if d['vs_window_B_robust'] else 'ns'}); "
        f"vs static-0.8 {d['vs_static_08']} "
        f"({'ROBUST' if d['vs_static_08_robust'] else 'ns'}); "
        f"win {d['win_rate']:.0%}"
        for d in diff_rows
    ]
    journal.summary(
        objective="Confirm the CUSUM adaptive-fusion win generalises across drift "
        "scenarios (sudden/gradual/recurring), robust across injection seeds.",
        method=f"CUSUM enter={enter:g} (calibrated once on control); coherent drift; "
        "paired proposed-baseline differences with 95% CIs, per scenario.",
        findings=findings,
        limitations=["Dense cohort; one trained backbone seed."],
        next_steps=["Backbone-seed variance; stronger long-term encoder; full scale."],
    )
    print(f"\nCUSUM enter={enter:g}")
    for d in diff_rows:
        w_tag = "ROBUST" if d["vs_window_B_robust"] else "ns"
        s_tag = "ROBUST" if d["vs_static_08_robust"] else "ns"
        print(
            f"  [{d['scenario']:<9}] vs window-B {d['vs_window_B']} {w_tag:>6} | "
            f"vs static-0.8 {d['vs_static_08']} {s_tag:>6} | win {d['win_rate']:.0%}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
