#!/usr/bin/env python3
"""Plot total operations with solid minima and hatched min-to-max extensions.

Totals include shared prep, specific prep, and inference. Ranges are over
the evaluated 4-, 9-, and 20-RR-interval cases in operation_counts.json;
they are not worst-case bounds over arbitrary ECGs. TinyAF denotes TinyAF-S.
Solid bars end at the minimum; translucent, diagonally hatched extensions end
at a solid maximum line. Variable counts have two-line max-to-min labels;
fixed counts use solid bars and a single numerical label.

Running this script writes ``operations.png`` and ``operations.pdf`` to
``fig/outputs``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator


METHODS = ("HR+HRV", "TinyAF", "SmallNN", "LargeNN")
MIN_OPERATIONS = (59_350, 72_179, 1_247_994, 436_398_742)
MAX_OPERATIONS = (62_540, 102_905, 1_247_994, 436_398_742)
LABELS = (
    r"$6.25 \times 10^{4}$ -" + "\n" + r"$5.94 \times 10^{4}$",
    r"$1.03 \times 10^{5}$ -" + "\n" + r"$7.22 \times 10^{4}$",
    r"$1.25 \times 10^{6}$",
    r"$4.36 \times 10^{8}$",
)

TINYAF_COLOR = "#1583F2"
GREY = "#9E9E9E"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def configure_style() -> None:
    """Configure consistent, publication-friendly typography."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 16,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure() -> tuple[plt.Figure, plt.Axes]:
    """Build and return the operations bar plot."""
    configure_style()
    fig, axis = plt.subplots(figsize=(7.2, 5.0))

    colors = [GREY, TINYAF_COLOR, GREY, GREY]
    bars = axis.bar(
        METHODS,
        MIN_OPERATIONS,
        width=0.68,
        color=colors,
        edgecolor=colors,
        linewidth=0.8,
        zorder=2,
    )

    for bar, minimum, maximum, color in zip(
        bars, MIN_OPERATIONS, MAX_OPERATIONS, colors
    ):
        if maximum > minimum:
            left, width = bar.get_x(), bar.get_width()
            axis.add_patch(Rectangle(
                (left, minimum), width, maximum - minimum,
                facecolor=to_rgba(color, 0.18), edgecolor=to_rgba(color, 0.65),
                hatch="///", linewidth=bar.get_linewidth(), zorder=3,
            ))
            axis.hlines(
                maximum, left, left + width,
                color=color, linewidth=1.2, capstyle="butt", zorder=4,
            )

    axis.set_yscale("log")
    axis.set_ylim(5.0e3, 1.2e9)
    axis.set_ylabel("Operations per 10s ECG", labelpad=8)
    axis.yaxis.set_minor_locator(NullLocator())
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", length=0, pad=8)

    for bar, label, value in zip(bars, LABELS, MAX_OPERATIONS):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.22,
            label,
            ha="center",
            multialignment="left",
            va="bottom",
            fontsize=13,
        )

    fig.tight_layout()
    return fig, axis


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, _ = make_figure()
    for extension in ("pdf", "png"):
        fig.savefig(
            OUTPUT_DIR / f"operations.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
