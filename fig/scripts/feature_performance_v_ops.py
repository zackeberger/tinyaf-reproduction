#!/usr/bin/env python3
"""Plot diagnostic performance of cardiac feature sets in BWH-Test.

Running this file writes separate AUROC and AUPRC dot-and-whisker panels,
including 95% confidence intervals, to ``fig/outputs`` in PNG and PDF formats.
"""

from pathlib import Path

import matplotlib.pyplot as plt


FEATURES = (
    {
        "name": "HR+HRV",
        "auroc": 0.857,
        "auroc_ci": (0.853, 0.861),
        "auprc": 0.379,
        "auprc_ci": (0.371, 0.389),
    },
    {
        "name": "RR Intervals",
        "auroc": None,
        "auroc_ci": None,
        "auprc": None,
        "auprc_ci": None,
    },
    {
        "name": "Aligned ECG",
        "auroc": 0.880,
        "auroc_ci": (0.878, 0.883),
        "auprc": 0.367,
        "auprc_ci": (0.359, 0.376),
    },
    {
        "name": "Aligned ECG +\nRR Intervals",
        "auroc": None,
        "auroc_ci": None,
        "auprc": None,
        "auprc_ci": None,
    },
)

AUROC_COLOR = "#2F75B5"
AUPRC_COLOR = "#E89C75"
GRID_COLOR = "#D8D8D8"
METRIC_ORDER = ("auroc", "auprc")
METRIC_COLORS = {"auroc": AUROC_COLOR, "auprc": AUPRC_COLOR}
METRIC_LIMITS = {"auroc": (0.80, 0.90), "auprc": (0.30, 0.40)}
METRIC_TICKS = {
    "auroc": (0.80, 0.85, 0.90),
    "auprc": (0.30, 0.35, 0.40),
}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def configure_style() -> None:
    """Match the typography and styling of the model-performance figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_metric_result(
    axis: plt.Axes,
    *,
    metric: str,
    y_position: float,
    value: float,
    interval: tuple[float, float],
) -> None:
    """Draw one point estimate with its horizontal 95% confidence interval."""
    color = METRIC_COLORS[metric]
    lower, upper = interval
    axis.errorbar(
        value,
        y_position,
        xerr=[[value - lower], [upper - value]],
        fmt="o",
        markersize=8.5,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=1.2,
        ecolor=color,
        elinewidth=2.0,
        capsize=5.0,
        capthick=2.0,
        zorder=3,
    )


def make_figure() -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Build and return the separate feature-performance metric panels."""
    configure_style()
    fig, axes = plt.subplots(ncols=2, figsize=(10.8, 4.4), sharey=True)
    method_positions = tuple(range(len(FEATURES)))

    for axis, metric in zip(axes, METRIC_ORDER):
        for position, feature in zip(method_positions, FEATURES):
            value = feature[metric]
            interval = feature[f"{metric}_ci"]
            if value is None:
                continue
            plot_metric_result(
                axis,
                metric=metric,
                y_position=position,
                value=value,
                interval=interval,
            )

        axis.set_xlim(*METRIC_LIMITS[metric])
        axis.set_xticks(METRIC_TICKS[metric])
        axis.set_title(metric.upper(), fontsize=19, fontweight="normal", pad=12)
        axis.set_ylim(len(FEATURES) - 0.5, -0.5)
        axis.grid(axis="x", color=GRID_COLOR, linewidth=0.75, alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", direction="out", pad=8)
        axis.tick_params(axis="y", length=0, pad=12)

    axes[0].set_yticks(
        method_positions,
        [feature["name"] for feature in FEATURES],
    )
    axes[1].tick_params(axis="y", labelleft=False)

    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(wspace=0.24)
    return fig, (axes[0], axes[1])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, _ = make_figure()
    for extension in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"feature_performance.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
