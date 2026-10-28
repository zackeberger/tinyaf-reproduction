#!/usr/bin/env python3
"""Plot AUROC and AUPRC against model operations.

Operation counts are total operations per 10-second ECG segment at L=9 RR
intervals from ``fig/util/operation_counter.tex``, including shared prep,
specific prep, and inference. TinyAF denotes TinyAF-S. Central performance
estimates and 95% confidence intervals come from the comparison table. Running
this file writes PNG and PDF versions of the figure to ``fig/outputs``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter


MODELS = (
    {
        "name": "HR+HRV",
        "operations": 60_562,
        "auroc": 0.8661,
        "auroc_ci": (0.8626, 0.8696),
        "auprc": 0.4044,
        "auprc_ci": (0.3956, 0.4141),
        "marker": "o",
    },
    {
        "name": "TinyAF",
        "operations": 81_844,
        "auroc": 0.9869,
        "auroc_ci": (0.9862, 0.9875),
        "auprc": 0.9160,
        "auprc_ci": (0.9116, 0.9197),
        "marker": "*",
    },
    {
        "name": "SmallNN",
        "operations": 1_247_994,
        "auroc": 0.9856,
        "auroc_ci": (0.9849, 0.9862),
        "auprc": 0.9021,
        "auprc_ci": (0.8974, 0.9062),
        "marker": "s",
    },
    {
        "name": "LargeNN",
        "operations": 436_398_742,
        "auroc": 0.9923,
        "auroc_ci": (0.9918, 0.9927),
        "auprc": 0.9529,
        "auprc_ci": (0.9503, 0.9553),
        "marker": "^",
    },
)

AUROC_COLOR = "#2F75B5"
AUPRC_COLOR = "#E89C75"
TEXT_COLOR = "#2B2B2B"
GRID_COLOR = "#D8D8D8"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def configure_style() -> None:
    """Use the same DejaVu Sans family embedded in the reference figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Build and return the dual-axis performance figure."""
    configure_style()
    fig, ax_auroc = plt.subplots(figsize=(6.8, 5.4))
    ax_auprc = ax_auroc.twinx()

    operations = [model["operations"] for model in MODELS]
    ax_auroc.plot(
        operations,
        [model["auroc"] for model in MODELS],
        color=AUROC_COLOR,
        linewidth=1.6,
        linestyle=":",
        alpha=0.4,
        zorder=1,
    )
    ax_auprc.plot(
        operations,
        [model["auprc"] for model in MODELS],
        color=AUPRC_COLOR,
        linewidth=1.6,
        linestyle=":",
        alpha=0.4,
        zorder=1,
    )

    for model in MODELS:
        marker_size = 255 if model["name"] == "TinyAF" else 115
        for axis, metric, color in (
            (ax_auroc, "auroc", AUROC_COLOR),
            (ax_auprc, "auprc", AUPRC_COLOR),
        ):
            axis.vlines(
                model["operations"],
                model[f"{metric}_ci"][0],
                model[f"{metric}_ci"][1],
                color=color,
                linewidth=5,
                alpha=0.22,
                zorder=2,
            )
            if model["name"] == "TinyAF":
                axis.scatter(
                    model["operations"],
                    model[metric],
                    s=440,
                    marker="o",
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.16,
                    clip_on=False,
                    zorder=3,
                )
            axis.scatter(
                model["operations"],
                model[metric],
                s=marker_size,
                marker=model["marker"],
                facecolor=color,
                edgecolor="white",
                linewidth=0.9,
                clip_on=False,
                zorder=4,
            )

    ax_auroc.set_xscale("log")
    ax_auroc.set_xlim(3.0e4, 1.0e9)
    ax_auroc.set_ylim(0.5, 1.03)
    ax_auprc.set_ylim(0.0, 1.03)
    ax_auroc.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_auprc.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax_auroc.set_xlabel("Operations")
    ax_auroc.set_ylabel("AUROC", color=AUROC_COLOR)
    ax_auprc.set_ylabel("AUPRC", color=AUPRC_COLOR)
    ax_auroc.set_title(
        "Performance in BWH-Test",
        fontsize=20,
        fontweight="normal",
        pad=12,
    )
    ax_auroc.tick_params(axis="y", colors=AUROC_COLOR)
    ax_auprc.tick_params(axis="y", colors=AUPRC_COLOR)
    ax_auroc.tick_params(axis="both", which="both", direction="out")
    ax_auprc.tick_params(axis="y", which="both", direction="out", pad=9)

    ax_auroc.xaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax_auroc.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=(2.0, 5.0), numticks=14)
    )
    ax_auroc.xaxis.set_minor_formatter(NullFormatter())
    ax_auroc.grid(axis="y", color=GRID_COLOR, linewidth=0.75, alpha=0.75)
    ax_auroc.set_axisbelow(True)

    ax_auroc.spines["top"].set_visible(False)
    ax_auprc.spines["top"].set_visible(False)
    ax_auroc.spines["left"].set_color(AUROC_COLOR)
    ax_auprc.spines["right"].set_color(AUPRC_COLOR)

    legend_order = (
        "LargeNN",
        "SmallNN",
        "TinyAF",
        "HR+HRV",
    )
    models_by_name = {model["name"]: model for model in MODELS}
    legend_models = [models_by_name[name] for name in legend_order]
    metric_handles = [
        Patch(facecolor=AUROC_COLOR, edgecolor="none", label="AUROC"),
        Patch(facecolor=AUPRC_COLOR, edgecolor="none", label="AUPRC"),
    ]
    model_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=model["marker"],
            markersize=10 if model["name"] == "TinyAF" else 8,
            markerfacecolor=TEXT_COLOR,
            markeredgecolor=TEXT_COLOR,
            markeredgewidth=1.0,
        )
        for model in legend_models
    ]

    metric_legend = ax_auroc.legend(
        handles=metric_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.05),
        ncol=1,
        frameon=False,
        fontsize=15,
        handlelength=1.0,
        handletextpad=0.45,
        borderaxespad=0.8,
    )
    ax_auroc.add_artist(metric_legend)
    ax_auroc.legend(
        handles=model_handles,
        labels=[model["name"] for model in legend_models],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        ncol=1,
        frameon=False,
        fontsize=15,
        handlelength=1.0,
        handletextpad=0.45,
        borderaxespad=0.8,
    )

    fig.tight_layout(pad=1.1)
    return fig, ax_auroc, ax_auprc


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, _, _ = make_figure()
    for extension in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"model_performance_v_ops.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
