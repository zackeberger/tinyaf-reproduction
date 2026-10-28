"""Plot hardcoded external-validation AUROC and AUPRC with confidence intervals.

Formatting follows external_val.py; no W&B connection or external data is needed.
Running this file writes external_validation_auroc and external_validation_auprc
as PNG and PDF files under fig/outputs (or --output-dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
DATASETS = ("SHDB-AF", "IRIDIA-AF", "CPSC2021")
MODEL_ORDER = ("HR+HRV", "TinyAF", "SmallNN", "LargeNN")
MODEL_COLORS = {
    "HR+HRV": "#9E9E9E",
    "TinyAF": "#1583F2",
    "SmallNN": "#9467BD",
    "LargeNN": "#D14949",
}
PLOT_MARKER_SIZE = 9.5

# Each tuple is (estimate, lower 95% CI bound, upper 95% CI bound).
# CI entries are absolute bounds, not error-bar widths.
# MicroAF-S results are displayed under this repository's TinyAF name.
AUROC = {
    "HR+HRV": {
        "SHDB-AF": (0.8633, 0.8120, 0.9048),
        "IRIDIA-AF": (0.9392, 0.9195, 0.9544),
        "CPSC2021": (0.7754, 0.7489, 0.8015),
    },
    "TinyAF": {
        "SHDB-AF": (0.9893, 0.9793, 0.9962),
        "IRIDIA-AF": (0.9939, 0.9908, 0.9962),
        "CPSC2021": (0.9864, 0.9792, 0.9921),
    },
    "SmallNN": {
        "SHDB-AF": (0.9918, 0.9842, 0.9973),
        "IRIDIA-AF": (0.9944, 0.9915, 0.9965),
        "CPSC2021": (0.9883, 0.9821, 0.9933),
    },
    "LargeNN": {
        "SHDB-AF": (0.9925, 0.9844, 0.9985),
        "IRIDIA-AF": (0.9956, 0.9930, 0.9974),
        "CPSC2021": (0.9940, 0.9896, 0.9974),
    },
}
AUPRC = {
    "HR+HRV": {
        "SHDB-AF": (0.5435, 0.4243, 0.6654),
        "IRIDIA-AF": (0.7232, 0.6600, 0.7819),
        "CPSC2021": (0.5258, 0.4584, 0.5871),
    },
    "TinyAF": {
        "SHDB-AF": (0.9357, 0.8727, 0.9847),
        "IRIDIA-AF": (0.9870, 0.9798, 0.9917),
        "CPSC2021": (0.9516, 0.9157, 0.9770),
    },
    "SmallNN": {
        "SHDB-AF": (0.9599, 0.9148, 0.9909),
        "IRIDIA-AF": (0.9887, 0.9827, 0.9926),
        "CPSC2021": (0.9725, 0.9546, 0.9853),
    },
    "LargeNN": {
        "SHDB-AF": (0.9641, 0.9181, 0.9948),
        "IRIDIA-AF": (0.9915, 0.9866, 0.9949),
        "CPSC2021": (0.9848, 0.9706, 0.9941),
    },
}


def configure_style() -> None:
    """Match the typography of external_val.py."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 18,
            "legend.fontsize": 15,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_panel(
    values: dict,
    *,
    ylabel: str,
    limits: tuple[float, float],
    tick_step: float,
) -> tuple[plt.Figure, plt.Axes]:
    """Build one panel from estimates and absolute confidence-interval bounds."""
    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    positions = np.arange(len(DATASETS))

    for index in range(len(DATASETS)):
        if index % 2:
            axis.axhspan(index - 0.5, index + 0.5, facecolor="#F0F0F0", zorder=0)

    offsets = np.linspace(-0.3, 0.3, len(MODEL_ORDER))
    for model_index, model in enumerate(MODEL_ORDER):
        means, low_errors, high_errors = [], [], []
        for dataset in DATASETS:
            mean, low, high = values[model][dataset]
            if not 0 <= low <= mean <= high <= 1:
                raise ValueError(f"Invalid {ylabel} estimate/CI for {model} on {dataset}")
            means.append(mean)
            low_errors.append(mean - low)
            high_errors.append(high - mean)
        color = MODEL_COLORS[model]
        line_width = 2.4 if model == "TinyAF" else 1.7
        y_positions = positions + offsets[model_index]

        axis.errorbar(
            means,
            y_positions,
            xerr=[low_errors, high_errors],
            fmt="none",
            ecolor=color,
            capsize=4,
            capthick=line_width,
            elinewidth=line_width,
            zorder=3,
        )
        axis.plot(
            means,
            y_positions,
            marker="o",
            color=color,
            linestyle="None",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=PLOT_MARKER_SIZE,
            zorder=4,
        )

    axis.set_xlim(limits[0] - 0.03, limits[1] + 0.03)
    axis.set_xticks(np.arange(limits[0], limits[1] + 0.001, tick_step))
    axis.set_xlabel(ylabel, fontsize=20)
    axis.set_yticks(positions, DATASETS, fontsize=18)
    axis.set_ylim(-0.5, len(DATASETS) - 0.5)
    axis.invert_yaxis()
    axis.tick_params(axis="x", labelsize=16)
    axis.tick_params(axis="y", length=0)
    axis.spines[["top", "right"]].set_visible(False)

    handles = [
        mlines.Line2D(
            [], [], color=MODEL_COLORS[model], marker="o", linestyle="None",
            markersize=PLOT_MARKER_SIZE, markeredgecolor="white", label=model,
        )
        for model in MODEL_ORDER
    ]
    axis.legend(
        handles=handles, loc="lower left", frameon=False, fontsize=15, handletextpad=0.6,
    )
    fig.tight_layout()
    return fig, axis


def save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    """Write one figure as PDF and PNG, then close it."""
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{stem}.{extension}", dpi=300,
            bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    for metric, values, limits, step in (
        ("AUROC", AUROC, (0.5, 1.0), 0.1),
        ("AUPRC", AUPRC, (0.0, 1.0), 0.2),
    ):
        fig, _ = make_panel(values, ylabel=metric, limits=limits, tick_step=step)
        save_figure(fig, f"external_validation_{metric.lower()}", args.output_dir)


if __name__ == "__main__":
    main()
