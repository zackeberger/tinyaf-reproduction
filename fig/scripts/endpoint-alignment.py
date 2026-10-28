#!/usr/bin/env python3
"""Plot endpoint-alignment performance for SmallNN and LargeNN.

The dumbbell plot compares models trained on AF with models trained on AF/AFL,
all scored on AF/AFL. Points include horizontal 95% confidence intervals where
available. The figure is written to ``fig/outputs`` in PNG and PDF formats.
"""

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt


RESULTS = {
    "SmallNN": {
        "PPV": {
            "Train on AF": (0.7687, 0.7605, 0.7770),
            "Train on AF/AFL": (0.8500, 0.8442, 0.8560),
        },
        "AUROC": {
            "Train on AF": (0.9782, 0.9772, 0.9791),
            "Train on AF/AFL": (0.9850, 0.9843, 0.9856),
        },
        "AUPRC": {
            "Train on AF": (0.8563, 0.8497, 0.8626),
            "Train on AF/AFL": (0.8961, 0.8912, 0.9009),
        },
    },
    "LargeNN": {
        "PPV": {
            "Train on AF": (0.9165, 0.9124, 0.9207),
            "Train on AF/AFL": (0.9518, 0.9489, 0.9546),
        },
        "AUROC": {
            "Train on AF": (0.9858, 0.9848, 0.9867),
            "Train on AF/AFL": (0.9915, 0.9911, 0.9920),
        },
        "AUPRC": {
            "Train on AF": (0.9253, 0.9214, 0.9295),
            "Train on AF/AFL": (0.9485, 0.9456, 0.9510),
        },
    },
}

MODEL_ORDER = ("SmallNN", "LargeNN")
METRIC_ORDER = ("AUROC", "AUPRC", "PPV")
TRAINING_ORDER = ("Train on AF", "Train on AF/AFL")
METRIC_COLORS = {"PPV": "#7A5195", "AUROC": "#2F75B5", "AUPRC": "#E89C75"}
TRAINING_MARKERS = {"Train on AF": "H", "Train on AF/AFL": "D"}
TRAINING_MARKER_SIZES = {"Train on AF": 7.5, "Train on AF/AFL": 6.5}
CONNECTOR_COLOR = "#B8B8B8"
METRIC_OFFSETS = {"AUROC": 0.24, "AUPRC": 0.0, "PPV": -0.24}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def configure_style() -> None:
    """Configure typography consistent with the other performance figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 18,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_result(
    axis: plt.Axes,
    *,
    y_position: float,
    color: str,
    train_on_af: tuple[float, float | None, float | None],
    train_on_af_afl: tuple[float, float | None, float | None],
) -> None:
    """Draw one dumbbell, adding confidence intervals when available."""
    af_value, af_low, af_high = train_on_af
    aligned_value, aligned_low, aligned_high = train_on_af_afl

    axis.plot(
        [af_value, aligned_value],
        [y_position, y_position],
        color=CONNECTOR_COLOR,
        linewidth=2.8,
        zorder=2,
    )

    for value, low, high, training_label in (
        (af_value, af_low, af_high, TRAINING_ORDER[0]),
        (aligned_value, aligned_low, aligned_high, TRAINING_ORDER[1]),
    ):
        errorbar_options = {}
        if low is not None and high is not None:
            errorbar_options = {
                "xerr": [[value - low], [high - value]],
                "ecolor": color,
                "elinewidth": 1.7,
                "capsize": 4.5,
                "capthick": 1.7,
            }

        axis.errorbar(
            value,
            y_position,
            fmt=TRAINING_MARKERS[training_label],
            markersize=TRAINING_MARKER_SIZES[training_label],
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=2.0,
            zorder=3,
            **errorbar_options,
        )


def make_figure() -> tuple[plt.Figure, plt.Axes]:
    """Build and return the endpoint-alignment dumbbell plot."""
    configure_style()
    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    model_positions = {model: 1 - index for index, model in enumerate(MODEL_ORDER)}

    for model in MODEL_ORDER:
        for metric in METRIC_ORDER:
            if metric not in RESULTS[model]:
                continue
            result = RESULTS[model][metric]
            plot_result(
                axis,
                y_position=model_positions[model] + METRIC_OFFSETS[metric],
                color=METRIC_COLORS[metric],
                train_on_af=result[TRAINING_ORDER[0]],
                train_on_af_afl=result[TRAINING_ORDER[1]],
            )

    axis.set_xlim(0.75, 1.0)
    axis.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    axis.set_title(
        "Performance on AF/AFL Endpoint; BWH-Test",
        fontsize=20,
        pad=18,
    )
    axis.set_yticks(
        [model_positions[model] for model in MODEL_ORDER],
        MODEL_ORDER,
    )
    axis.set_ylim(-0.45, 1.45)
    axis.grid(axis="x", color="#D8D8D8", linewidth=0.75, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", pad=8)
    axis.tick_params(axis="y", length=0, pad=10)

    metric_legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=METRIC_COLORS[metric],
            linewidth=2.8,
            linestyle="-",
            label=metric,
        )
        for metric in METRIC_ORDER
    ]
    training_legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="#A5A5A5",
            marker=TRAINING_MARKERS[TRAINING_ORDER[0]],
            markerfacecolor="#A5A5A5",
            markeredgewidth=1.8,
            linestyle="None",
            markersize=TRAINING_MARKER_SIZES[TRAINING_ORDER[0]],
            label=TRAINING_ORDER[0],
        ),
        mlines.Line2D(
            [],
            [],
            color="#A5A5A5",
            marker=TRAINING_MARKERS[TRAINING_ORDER[1]],
            markerfacecolor="#A5A5A5",
            linestyle="None",
            markersize=TRAINING_MARKER_SIZES[TRAINING_ORDER[1]],
            label=TRAINING_ORDER[1],
        ),
    ]
    metric_legend = axis.legend(
        handles=metric_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
        fontsize=17,
        columnspacing=1.8,
        handletextpad=0.7,
    )
    axis.add_artist(metric_legend)
    axis.legend(
        handles=training_legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        ncol=1,
        frameon=False,
        fontsize=17,
        labelspacing=0.9,
        handletextpad=0.7,
    )

    fig.tight_layout()
    return fig, axis


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, _ = make_figure()
    for extension in ("pdf", "png"):
        fig.savefig(
            OUTPUT_DIR / f"endpoint-alignment.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
