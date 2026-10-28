#!/usr/bin/env python3
"""Generate paired TinyAF predictive-value bars with confidence intervals.

Running the script writes ``predictive_value.pdf`` and
``predictive_value.png`` to ``fig/outputs`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# PPV and NPV are (point estimate, lower 95% CI, upper 95% CI). Prevalence is
# shown as a point estimate only. Values are proportions in [0, 1].
# User-supplied MicroAF-S results; MicroAF-S is equivalent to TinyAF.
RESULTS = (
    (
        "SHDB-AF",
        0.2112,  # 176,983 / 837,854
        (0.9020, 0.8358, 0.9506),
        (0.9814, 0.9667, 0.9929),
    ),
    (
        "IRIDIA-AF",
        0.2405,  # 578,833 / 2,406,294
        (0.9662, 0.9514, 0.9779),
        (0.9813, 0.9748, 0.9874),
    ),
    (
        "CPSC2021",
        0.3429,  # 59,026 / 172,142
        (0.9349, 0.9032, 0.9558),
        (0.9667, 0.9566, 0.9762),
    ),
)

TINYAF_BLUE = "#1583F2"
NPV_BLUE = "#0B5FA5"
TEXT_COLOR = "#202020"


def configure_style() -> None:
    """Match the typography used by the other TinyAF figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_metric_bar(
    axis: plt.Axes,
    *,
    y: float,
    result: tuple[float, float, float],
    color: str,
) -> None:
    """Draw one horizontal estimate bar with a black 95% CI whisker."""
    value, lower, upper = result
    if not 0.0 <= lower <= value <= upper <= 1.0:
        raise ValueError(
            "Metric values must satisfy 0 <= lower <= estimate <= upper <= 1; "
            f"got {result}"
        )

    axis.barh(
        y,
        value,
        height=0.28,
        color=color,
        edgecolor="none",
        zorder=2,
    )
    axis.errorbar(
        value,
        y,
        xerr=[[value - lower], [upper - value]],
        fmt="none",
        ecolor="#111111",
        elinewidth=2.0,
        capsize=5.0,
        capthick=2.0,
        zorder=4,
    )
    axis.scatter(
        [value],
        [y],
        s=24,
        facecolor="#111111",
        edgecolor="#111111",
        linewidth=0,
        zorder=5,
    )


def make_figure() -> tuple[plt.Figure, plt.Axes]:
    """Build paired horizontal PPV/NPV bars for each external dataset."""
    configure_style()
    fig, axis = plt.subplots(figsize=(8.2, 4.3))
    axis.set_xlim(0, 1)
    axis.set_ylim(0.0, 6.05)
    group_centers = (4.85, 2.95, 1.05)
    metric_offset = 0.27

    prevalence_handle = Line2D(
        [],
        [],
        marker="D",
        linestyle="None",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="#111111",
        markeredgewidth=0.9,
        label="AF/AFL prevalence",
    )
    ppv_handle = Patch(facecolor=TINYAF_BLUE, edgecolor="none", label="PPV")
    npv_handle = Patch(facecolor=NPV_BLUE, edgecolor="none", label="NPV")
    axis.legend(
        handles=[ppv_handle, npv_handle, prevalence_handle],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.005),
        ncol=3,
        borderaxespad=0,
        frameon=False,
        fontsize=14,
        columnspacing=1.3,
        handlelength=1.3,
        handletextpad=0.6,
    )

    for (dataset, prevalence, ppv, npv), center in zip(RESULTS, group_centers):
        axis.text(
            -0.035,
            center,
            dataset,
            ha="right",
            va="center",
            fontsize=18,
            color=TEXT_COLOR,
            clip_on=False,
            zorder=5,
        )

        draw_metric_bar(
            axis,
            y=center + metric_offset,
            result=ppv,
            color=TINYAF_BLUE,
        )
        axis.scatter(
            [prevalence],
            [center + metric_offset],
            marker="D",
            s=38,
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.9,
            zorder=6,
        )
        draw_metric_bar(
            axis,
            y=center - metric_offset,
            result=npv,
            color=NPV_BLUE,
        )

    axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    axis.set_yticks([])
    axis.grid(axis="x", color="#D4D4D4", linewidth=0.8, zorder=1)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.9)
    axis.tick_params(axis="x", labelsize=14, pad=6)
    axis.tick_params(axis="y", length=0)

    fig.subplots_adjust(left=0.24, right=0.975, top=0.975, bottom=0.15)
    return fig, axis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--stem", default="predictive_value")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = make_figure()
    for extension in ("pdf", "png"):
        fig.savefig(
            args.output_dir / f"{args.stem}.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
