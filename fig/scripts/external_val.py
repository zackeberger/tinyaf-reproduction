#!/usr/bin/env python3
"""Generate separate AF/AFL external-validation AUROC and AUPRC plots.

The underlying data and confidence intervals match ``figure3_profile_both``.
W&B is accessed read-only, and the resulting figures are written to
``fig/outputs`` as ``external_validation_auroc`` and
``external_validation_auprc``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from performance_profiles import FULL_PROJECT, extract_plot_data, fetch_runs


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# Keys match the model names returned by performance_profiles.extract_plot_data.
MODEL_ORDER = ("HR+HRV", "TinyAF", "Kim 2022", "Ribeiro 2020")
MODEL_LABELS = {
    "HR+HRV": "HR+HRV-AF/AFL",
    "TinyAF": "TinyAF/AFL",
    "Kim 2022": "SmallNN-AF/AFL",
    "Ribeiro 2020": "LargeNN-AF/AFL",
}
MODEL_COLORS = {
    "HR+HRV": "#9E9E9E",
    "TinyAF": "#1583F2",
    "Kim 2022": "#9467BD",
    "Ribeiro 2020": "#D14949",
}
MODEL_MARKERS = {
    "HR+HRV": "o",
    "TinyAF": "o",
    "Kim 2022": "o",
    "Ribeiro 2020": "o",
}
MODEL_MARKER_SIZES = {
    "HR+HRV": 9.5,
    "TinyAF": 9.5,
    "Kim 2022": 9.5,
    "Ribeiro 2020": 9.5,
}
PLOT_MARKER_SIZE = 9.5


def configure_style() -> None:
    """Match the typography of the source external-validation profile."""
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
    datasets: list[str],
    *,
    ylabel: str,
    limits: tuple[float, float],
    tick_step: float,
) -> tuple[plt.Figure, plt.Axes]:
    """Build one external-validation performance panel."""
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    positions = np.arange(len(datasets))

    for index in range(len(datasets)):
        if index % 2:
            axis.axhspan(
                index - 0.5,
                index + 0.5,
                facecolor="#F0F0F0",
                zorder=0,
            )

    offsets = np.linspace(-0.3, 0.3, len(MODEL_ORDER))
    for model_index, model in enumerate(MODEL_ORDER):
        points = [values[model].get(dataset) for dataset in datasets]
        means = [point[0] if point else np.nan for point in points]
        low_errors = [point[1] if point else np.nan for point in points]
        high_errors = [point[2] if point else np.nan for point in points]
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
            marker=MODEL_MARKERS[model],
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
    axis.set_yticks(positions, datasets, fontsize=18)
    axis.set_ylim(-0.5, len(datasets) - 0.5)
    axis.invert_yaxis()
    axis.tick_params(axis="x", labelsize=16)
    axis.tick_params(axis="y", length=0)
    axis.spines[["top", "right"]].set_visible(False)

    handles = [
        mlines.Line2D(
            [],
            [],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linestyle="None",
            markersize=MODEL_MARKER_SIZES[model],
            markeredgecolor="white",
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    axis.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=15,
        handletextpad=0.6,
    )

    fig.tight_layout()
    return fig, axis


def save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    """Write one figure as PDF and PNG, then close it."""
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    frame = fetch_runs(FULL_PROJECT)
    auroc, auprc, datasets = extract_plot_data(frame, label="BOTH", endpoint=None)
    auroc_figure, _ = make_panel(
        auroc,
        datasets,
        ylabel="AUROC",
        limits=(0.5, 1.0),
        tick_step=0.1,
    )
    auprc_figure, _ = make_panel(
        auprc,
        datasets,
        ylabel="AUPRC",
        limits=(0.0, 1.0),
        tick_step=0.2,
    )
    save_figure(auroc_figure, "external_validation_auroc", args.output_dir)
    save_figure(auprc_figure, "external_validation_auprc", args.output_dir)


if __name__ == "__main__":
    main()
