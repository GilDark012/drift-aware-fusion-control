#!/usr/bin/env python3
"""Generate the original figures used in the PFE thesis.

Every figure is drawn from numbers produced by logged runs in this repository
(see the result documents under ``docs/``). A colour-blind-safe palette is used
throughout and all in-figure text is at least 9pt, per the aivancity guidelines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("output/figures")
# Colour-blind-safe (Okabe-Ito) palette.
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "figure.autolayout": True})


def architecture() -> None:
    """Pipeline: stream, backbone, signal, detector, policy, ranking."""
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.axis("off")
    stages = [
        ("Interaction\nstream", "#EAF2FA"),
        ("k-core +\nchronological\nsplit", "#EAF2FA"),
        ("Frozen backbone\nL(t) / S(t)", "#DCEAF7"),
        ("Drift signal\nD(t)", "#FDF0DC"),
        ("Detector\n(state machine)", "#FDF0DC"),
        ("Fusion policy\nalpha(t)", "#DFF2EA"),
        ("Ranked\nitems", "#DFF2EA"),
    ]
    x, w, gap = 0.2, 1.65, 0.42
    for index, (label, colour) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x, 1.05), w, 0.8, boxstyle="round,pad=0.06",
                                    fc=colour, ec="#333333", lw=1.1))
        ax.text(x + w / 2, 1.45, label, ha="center", va="center", fontsize=9)
        if index > 0:
            ax.add_patch(FancyArrowPatch((x - gap, 1.45), (x - 0.05, 1.45),
                                         arrowstyle="-|>", mutation_scale=12, lw=1.1,
                                         color="#333333"))
        x += w + gap
    ax.text(0.2 + 3 * (w + gap), 0.55,
            "trained once, then frozen: only alpha(t) changes downstream",
            ha="center", fontsize=8.5, style="italic", color=GREY)
    ax.set_xlim(0, x)
    ax.set_ylim(0.2, 2.3)
    fig.savefig(OUT / "fig_architecture.png", bbox_inches="tight")
    plt.close(fig)


def state_machine() -> None:
    """Detector states and the transitions between them."""
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.axis("off")
    nodes = {
        "STABLE": (1.0, 1.6, "#EAF2FA"),
        "EMERGING": (3.4, 1.6, "#FDF0DC"),
        "CONFIRMED": (5.8, 1.6, "#DFF2EA"),
        "TEMPORARY": (3.4, 0.4, "#F2E6F2"),
    }
    for name, (x, y, colour) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - 0.72, y - 0.28), 1.44, 0.56,
                                    boxstyle="round,pad=0.05", fc=colour,
                                    ec="#333333", lw=1.2))
        ax.text(x, y, name, ha="center", va="center", fontsize=9.5)

    def arrow(start: str, end: str, text: str) -> None:
        (x1, y1, _), (x2, y2, _) = nodes[start], nodes[end]
        ax.add_patch(FancyArrowPatch((x1 + 0.74, y1), (x2 - 0.74, y2),
                                     arrowstyle="-|>", mutation_scale=12, lw=1.1,
                                     color="#333333"))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", fontsize=8.2)

    arrow("STABLE", "EMERGING", "z >= z+")
    arrow("EMERGING", "CONFIRMED", "streak >= p")
    ax.add_patch(FancyArrowPatch((3.4, 1.3), (3.4, 0.7), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color="#333333"))
    ax.text(3.55, 1.0, "gap > g", fontsize=8.2)
    ax.add_patch(FancyArrowPatch((2.68, 0.4), (1.0, 1.30),
                                 connectionstyle="arc3,rad=-0.3", arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color="#333333"))
    ax.text(1.72, 0.50, "fades", fontsize=8.2)
    ax.add_patch(FancyArrowPatch((5.8, 1.90), (1.0, 1.90),
                                 connectionstyle="arc3,rad=0.32", arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color="#333333"))
    ax.text(3.4, 2.88, "z <= z- for r consecutive steps (recovery)",
            ha="center", fontsize=8.2)
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 3.1)
    fig.savefig(OUT / "fig_state_machine.png", bbox_inches="tight")
    plt.close(fig)


def alpha_response() -> None:
    """Alpha-response curves on the dense cohort for both long-term encoders."""
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    dense_val = [0.01834, 0.01960, 0.01946, 0.02153, 0.02346, 0.01985]
    dense_test = [0.01624, 0.01666, 0.01688, 0.01669, 0.01690, 0.01361]
    attention_val = [0.01828, 0.02000, 0.02143, 0.02084, 0.01893, 0.01208]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(alphas, dense_val, "o-", color=BLUE, lw=1.8,
            label="Recency pool - validation")
    ax.plot(alphas, dense_test, "s-", color=GREEN, lw=1.8, label="Recency pool - test")
    ax.plot(alphas, attention_val, "^--", color=ORANGE, lw=1.6,
            label="Attention pool - validation")
    ax.axvline(0.8, color=GREY, ls=":", lw=1.2)
    ax.annotate("optimum alpha = 0.8", xy=(0.8, 0.02346), xytext=(0.30, 0.02490),
                fontsize=9, arrowprops={"arrowstyle": "->", "color": GREY})
    ax.set_ylim(0.0110, 0.0262)
    ax.set_xlabel("Fusion weight alpha (0 = pure short-term, 1 = pure long-term)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Quality as a function of the fusion weight", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.savefig(OUT / "fig_alpha_response.png", bbox_inches="tight")
    plt.close(fig)


def adaptation_ci() -> None:
    """Paired differences (proposed minus baseline) with 95% CIs, per scenario."""
    scenarios = ["Sudden", "Gradual", "Recurring"]
    vs_window = [0.00078, 0.00165, 0.00069]
    err_window = [0.00010, 0.00019, 0.00008]
    vs_static = [0.00016, -0.00012, 0.00008]
    err_static = [0.00030, 0.00027, 0.00024]
    positions = list(range(len(scenarios)))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    offset = 0.16
    ax.errorbar(vs_window, [p + offset for p in positions], xerr=err_window, fmt="o",
                color=BLUE, capsize=4, lw=1.6, label="proposed - window-B")
    ax.errorbar(vs_static, [p - offset for p in positions], xerr=err_static, fmt="s",
                color=ORANGE, capsize=4, lw=1.6, label="proposed - static alpha=0.8")
    ax.axvline(0, color=VERM, ls="--", lw=1.2)
    ax.set_yticks(positions)
    ax.set_yticklabels(scenarios)
    ax.set_xlabel("Difference in overall NDCG@10 (95% CI, 30 injection seeds)")
    ax.set_title("Adaptive fusion against each fixed policy", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.savefig(OUT / "fig_adaptation_ci.png", bbox_inches="tight")
    plt.close(fig)


def encoder_mirror() -> None:
    """The encoder ablation: which fixed baseline is strong flips with the encoder."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    labels = ["static\nalpha=0.8", "window-B", "proposed"]
    recency = [0.01058, 0.00996, 0.01074]
    rec_err = [0.00038, 0.00034, 0.00033]
    attention = [0.00564, 0.01243, 0.01236]
    att_err = [0.00015, 0.00057, 0.00063]
    panels = (
        (axes[0], recency, rec_err, "Recency-pool long-term encoder"),
        (axes[1], attention, att_err, "Attention long-term encoder"),
    )
    for ax, values, errors, title in panels:
        bars = ax.bar(labels, values, yerr=errors, capsize=4,
                      color=[ORANGE, BLUE, GREEN], edgecolor="#333333", lw=0.8)
        best = max(range(len(values)), key=lambda i: values[i])
        bars[best].set_hatch("//")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Overall NDCG@10")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Which fixed policy is strong depends on the encoder (hatched = best)",
                 fontsize=10.5)
    fig.savefig(OUT / "fig_encoder_mirror.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Render every thesis figure into ``output/figures``."""
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    state_machine()
    alpha_response()
    adaptation_ci()
    encoder_mirror()
    for path in sorted(OUT.glob("fig_*.png")):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
