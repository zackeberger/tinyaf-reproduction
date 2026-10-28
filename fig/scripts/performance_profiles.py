#!/usr/bin/env python3
"""Generate all external-validation performance profiles from W&B.

W&B is accessed read-only. Set ``WANDB_API_KEY`` before running. All figures
are written as both PDF and PNG files under ``fig/outputs``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


ENTITY = "doprakah"
ENDPOINT_PROJECT = "afib_eval_endpoint_new"
FULL_PROJECT = "afib_eval_full_clean_new"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

MODEL_NAMES = {
    "RESNET_18_CNN": "Ribeiro 2020",
    "TINY_CNN": "Kim 2022",
    "F_HR+HRV_ALIGNED_CAT_TWO_CNN_60_40_16_3_1_8": "TinyAF",
    "NORM_HR+HRV_LINEAR": "HR+HRV",
}
MODEL_ORDER = ("Ribeiro 2020", "Kim 2022", "TinyAF", "HR+HRV")
MODEL_COLORS = {
    "TinyAF": "#B71C1C",
    "HR+HRV": "#9E9E9E",
    "Ribeiro 2020": "#1F77B4",
    "Kim 2022": "#9467BD",
}
DATASET_NAMES = {
    "MGH": "MGH",
    "ICENTIA": "Icentia11k",
    "CPSC": "CPSC2021",
    "SHDB": "SHDB-AF",
    "IRIDIA": "IRIDIA-AF",
}
DATASET_ORDER = ("MGH", "Icentia11k", "SHDB-AF", "IRIDIA-AF", "CPSC2021")
METRIC_COLUMNS = (
    "roc_auc_boot_mean", "roc_auc_ci_low", "roc_auc_ci_high",
    "pr_auc_boot_mean", "pr_auc_ci_low", "pr_auc_ci_high",
)


def fetch_runs(project: str) -> pd.DataFrame:
    """Return W&B run summaries/configs without modifying remote state."""
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY to the credential used by the notebook.")
    api = wandb.Api(timeout=90)
    records: list[dict] = []
    filters = {"config.model": {"$in": list(MODEL_NAMES)}}
    for run in api.runs(f"{ENTITY}/{project}", filters=filters, per_page=100, lazy=False):
        record = dict(run.config)
        record.update(dict(run.summary))
        records.append(record)
    if not records:
        raise RuntimeError(f"No runs found in {ENTITY}/{project}.")
    frame = pd.json_normalize(records)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    required = {"model", "data", "label", *METRIC_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{project} is missing required fields: {', '.join(missing)}")
    sort_column = "roc_auc" if "roc_auc" in frame else "roc_auc_boot_mean"
    dedupe = [c for c in ("model", "data", "label", "endpoint") if c in frame]
    return (
        frame.sort_values(sort_column, ascending=False)
        .drop_duplicates(subset=dedupe, keep="first")
        .reset_index(drop=True)
    )


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def extract_plot_data(
    frame: pd.DataFrame, label: str, endpoint: str | None,
) -> tuple[dict, dict, list[str]]:
    data = frame.copy()
    data.columns = data.columns.str.strip()
    data = data[data["label"].eq(label)]
    if endpoint is not None:
        if "endpoint" not in data:
            raise RuntimeError("The selected W&B project has no endpoint field.")
        data = data[data["endpoint"].eq(endpoint)]
    data = data[data["model"].isin(MODEL_NAMES)].copy()
    data["model_name"] = data["model"].map(MODEL_NAMES)
    data = data[~data["data"].eq("BWH")]
    data["data"] = data["data"].map(DATASET_NAMES).fillna(data["data"])
    aggregate = data.groupby(["model_name", "data"])[list(METRIC_COLUMNS)].mean().reset_index()
    if aggregate.empty:
        raise RuntimeError(f"No profile data found for label={label}, endpoint={endpoint}.")

    roc = {model: {} for model in MODEL_ORDER}
    pr = {model: {} for model in MODEL_ORDER}
    for row in aggregate.itertuples():
        roc[row.model_name][row.data] = (
            row.roc_auc_boot_mean,
            row.roc_auc_boot_mean - row.roc_auc_ci_low,
            row.roc_auc_ci_high - row.roc_auc_boot_mean,
        )
        pr[row.model_name][row.data] = (
            row.pr_auc_boot_mean,
            row.pr_auc_boot_mean - row.pr_auc_ci_low,
            row.pr_auc_ci_high - row.pr_auc_boot_mean,
        )
    present = set(aggregate["data"])
    datasets = [name for name in DATASET_ORDER if name in present]
    datasets.extend(sorted(present - set(datasets)))
    return roc, pr, datasets


def make_profile(
    frame: pd.DataFrame,
    *,
    label: str,
    endpoint: str | None,
    title: str,
    auc_limits: tuple[float, float],
) -> plt.Figure:
    roc, pr, datasets = extract_plot_data(frame, label, endpoint)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.2, 3.8))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02, color="#222222")

    def panel(ax: plt.Axes, values: dict, ylabel: str, limits: tuple[float, float], step: float) -> None:
        positions = np.arange(len(datasets))
        for index in range(len(datasets)):
            if index % 2:
                ax.axvspan(index - 0.5, index + 0.5, facecolor="#F0F0F0", zorder=0)
        offsets = np.linspace(-0.2, 0.2, len(MODEL_ORDER))
        for model_index, model in enumerate(MODEL_ORDER):
            points = [values[model].get(dataset) for dataset in datasets]
            means = [point[0] if point else np.nan for point in points]
            low = [point[1] if point else np.nan for point in points]
            high = [point[2] if point else np.nan for point in points]
            color = MODEL_COLORS[model]
            width = 1.8 if model == "TinyAF" else 1.0
            x = positions + offsets[model_index]
            ax.errorbar(x, means, yerr=[low, high], fmt="none", ecolor=color,
                        capsize=2, elinewidth=width * 0.8, zorder=3)
            ax.plot(x, means, marker="o", color=color, linestyle="None",
                    markeredgecolor="white", markeredgewidth=0.5,
                    markersize=5.5, zorder=4)
        ax.set_xticks(positions, datasets, fontsize=8, rotation=30, ha="right")
        ax.set_xlim(-0.5, len(datasets) - 0.5)
        ax.set_ylim(limits[0] - 0.03, limits[1] + 0.03)
        ax.set_yticks(np.arange(limits[0], limits[1] + 0.001, step))
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    panel(ax_roc, roc, "AUROC", auc_limits, 0.10)
    panel(ax_pr, pr, "AUPRC", (0.0, 1.0), 0.20)
    handles = [mlines.Line2D([], [], color=MODEL_COLORS[m], marker="o", linestyle="None",
                            markersize=5.5, markeredgecolor="white", label=m) for m in MODEL_ORDER]
    ax_roc.legend(handles=handles, loc="lower left", frameon=True, fontsize=7)
    ax_pr.legend(handles=handles, loc="upper right" if label == "FLUT" or endpoint == "FLUT" else "lower right",
                 frameon=True, fontsize=7)
    fig.subplots_adjust(wspace=0.28, top=0.93, bottom=0.22, left=0.10, right=0.96)
    return fig


def save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    endpoint_data = fetch_runs(ENDPOINT_PROJECT)
    for endpoint, name in (("AFIB", "AF"), ("FLUT", "AFL")):
        save_figure(
            make_profile(endpoint_data, label="BOTH", endpoint=endpoint,
                         title=f"External Validation of AF/AFL Detectors on {name} Endpoint",
                         auc_limits=(0.5, 1.0)),
            f"section8_profile_both_endpoint_{endpoint.lower()}", args.output_dir,
        )

    full_data = fetch_runs(FULL_PROJECT)
    variants = (
        ("BOTH", (0.5, 1.0), "Performance of AF/AFL Detectors on External Validation Data"),
        ("AFIB", (0.5, 1.0), "Performance of AF Detectors on External Validation Data"),
        ("FLUT", (0.0, 1.0), "Performance of AFL Detectors on External Validation Data"),
    )
    for label, limits, title in variants:
        save_figure(make_profile(full_data, label=label, endpoint=None, title=title,
                                 auc_limits=limits),
                    f"figure3_profile_{label.lower()}", args.output_dir)


if __name__ == "__main__":
    main()
