#!/usr/bin/env python3
"""Generate every Pareto-frontier variant from the notebook using W&B data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


ENTITY = "doprakah"
PROJECT = "afib_eval_full_clean_new"
TRAIN_PROJECT = "afib_train"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
BUCKET_EDGES = (1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6)
DATASETS = ("BWH", "MGH", "ICENTIA", "IRIDIA", "SHDB", "CPSC")
DATASET_NAMES = {"BWH": "BWH", "MGH": "MGH", "ICENTIA": "Icentia11k",
                 "IRIDIA": "IRIDIA-AF", "SHDB": "SHDB-AF", "CPSC": "CPSC2021"}
COLORS = {"Aligned ECG +HR/HRV": "#008C95", "Raw ECG": "#E85D75"}


def fetch_runs() -> pd.DataFrame:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY to the credential used by the notebook.")
    api = wandb.Api(timeout=90)
    records = []
    filters = {"config.model": {"$regex": "B.*"}}
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters=filters, per_page=100, lazy=False):
        record = dict(run.config)
        record.update(dict(run.summary))
        records.append(record)
    frame = pd.json_normalize(records)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    required = {"model", "fname", "data", "roc_auc_boot_mean", "roc_auc_ci_low",
                "roc_auc_ci_high", "pr_auc_boot_mean", "pr_auc_ci_low", "pr_auc_ci_high"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{PROJECT} is missing required fields: {', '.join(missing)}")
    sort_column = "roc_auc" if "roc_auc" in frame else "roc_auc_boot_mean"
    dedupe = [c for c in ("model", "data", "label", "endpoint") if c in frame]
    return frame.sort_values(sort_column, ascending=False).drop_duplicates(dedupe).reset_index(drop=True)


def _numeric_count(value: object) -> float:
    """Convert W&B numeric values, including strings such as ``34.954K``."""
    if value is None:
        return np.nan
    text = str(value).strip()
    multipliers = {"K": 1e3, "M": 1e6, "G": 1e9}
    multiplier = multipliers.get(text[-1:].upper(), 1.0)
    if multiplier != 1.0:
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def fetch_training_operations() -> pd.DataFrame:
    """Read measured model MACs and parameter counts from W&B read-only."""
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY to the credential used by the notebook.")
    api = wandb.Api(timeout=90)
    filters = {"config.model": {"$regex": "B.*"}}
    records = []
    for run in api.runs(
        f"{ENTITY}/{TRAIN_PROJECT}", filters=filters, per_page=100, lazy=False
    ):
        record = dict(run.config)
        record.update(dict(run.summary))
        records.append(record)
    frame = pd.json_normalize(records)
    required = {"model", "label", "macs_fvcore", "params_fvcore"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{TRAIN_PROJECT} is missing required fields: {', '.join(missing)}"
        )
    frame["model_macs"] = frame["macs_fvcore"].map(_numeric_count)
    frame["model_params"] = frame["params_fvcore"].map(_numeric_count)
    frame = frame.dropna(subset=["model_macs", "model_params"])
    return (
        frame.groupby(["model", "label"], as_index=False)[
            ["model_macs", "model_params"]
        ]
        .mean()
    )


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[frame["model"].str.startswith("B", na=False)].copy()
    extracted = data["fname"].str.extract(r"model_(.*?)_cal", expand=False)
    data["model_name"] = extracted.fillna(data["model"])
    data["bucket"] = pd.to_numeric(data["model_name"].str.extract(r"B(\d+)_", expand=False), errors="coerce")
    data["group"] = np.select(
        [data["model_name"].str.contains(r"HR\+HRV_ALIGNED", na=False),
         data["model_name"].str.contains("CAT_TWO_CNN", na=False)],
        ["Aligned ECG +HR/HRV", "Raw ECG"], default=None,
    )
    return data.dropna(subset=["bucket", "group"]).assign(bucket=lambda d: d["bucket"].astype(int))


def best_with_envelope(frame: pd.DataFrame, metric: str, low: str, high: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    best = frame.loc[frame.groupby(["group", "bucket"])[metric].idxmax()].copy()
    parts = []
    for _, group in best.groupby("group"):
        group = group.sort_values("bucket").copy()
        winners = group[metric].cummax().eq(group[metric])
        group["pareto"] = group[metric].cummax()
        group["pareto_low"] = group[low].where(winners).ffill()
        group["pareto_high"] = group[high].where(winners).ffill()
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def bucket_labels() -> list[str]:
    return [rf"$10^{{{int(np.log10(x))}}}$" if x in (1e3, 1e4, 1e5, 1e6)
            else rf"${int(x / 10 ** int(np.log10(x)))}\times10^{{{int(np.log10(x))}}}$"
            for x in BUCKET_EDGES]


def style(publication: bool = True) -> None:
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 9 if publication else 10, "axes.linewidth": 1.0,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def plot_axis(ax: plt.Axes, frame: pd.DataFrame, metric: str, low: str, high: str,
              metric_name: str, *, confidence: bool) -> None:
    envelope = best_with_envelope(prepare(frame), metric, low, high)
    for group_name, group in envelope.groupby("group"):
        group = group.sort_values("bucket")
        x = group["bucket"].to_numpy()
        color = COLORS[group_name]
        if confidence:
            ax.fill_between(x, group["pareto_low"], group["pareto_high"], color=color, alpha=0.15, edgecolor="none")
            ax.errorbar(x, group["pareto"],
                        yerr=[group["pareto"] - group["pareto_low"], group["pareto_high"] - group["pareto"]],
                        fmt="none", color=color, capsize=2, elinewidth=0.8, alpha=0.6)
        ax.plot(x, group["pareto"], "-o", color=color, label=group_name,
                markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks(range(len(BUCKET_EDGES)), bucket_labels(), rotation=45, ha="right")
    ax.set_xlim(-0.3, len(BUCKET_EDGES) - 0.7)
    ax.set_ylim(0.5 if metric_name == "AUROC" else 0.0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)


def save(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def simple_figures(frame: pd.DataFrame, output_dir: Path) -> None:
    for metric, low, high, name in (
        ("roc_auc_boot_mean", "roc_auc_ci_low", "roc_auc_ci_high", "AUROC"),
        ("pr_auc_boot_mean", "pr_auc_ci_low", "pr_auc_ci_high", "AUPRC"),
    ):
        style(False)
        fig, ax = plt.subplots()
        plot_axis(ax, frame, metric, low, high, name, confidence=False)
        ax.set_xlabel("Model FLOPs")
        ax.set_ylabel(name)
        ax.set_title(f"Pareto Frontier: {name} vs Compute")
        ax.legend()
        fig.tight_layout()
        save(fig, f"pareto_basic_{name.lower()}", output_dir)


def bwh_combined(frame: pd.DataFrame, output_dir: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    bwh = frame[frame["data"].eq("BWH")]
    for index, (metric, low, high, name) in enumerate((
        ("roc_auc_boot_mean", "roc_auc_ci_low", "roc_auc_ci_high", "AUROC"),
        ("pr_auc_boot_mean", "pr_auc_ci_low", "pr_auc_ci_high", "AUPRC"),
    )):
        plot_axis(axes[index], bwh, metric, low, high, name, confidence=True)
        axes[index].set_title(f"{name} vs Compute", fontweight="bold")
        axes[index].set_xlabel("Model FLOPs")
        axes[index].legend(frameon=True, loc="lower right", fontsize=7)
        axes[index].text(-0.2, 1.1, "ab"[index], transform=axes[index].transAxes,
                         fontsize=10, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save(fig, "pareto_with_ci", output_dir)


def _budget_frontier(
    frame: pd.DataFrame, metric: str, resource_column: str
) -> pd.DataFrame:
    """Return improving models, extended through the largest resource budget."""
    ordered = frame.sort_values([resource_column, metric], ascending=[True, False])
    ordered = ordered.drop_duplicates(resource_column, keep="first").copy()
    previous_best = ordered[metric].cummax().shift(fill_value=-np.inf)
    frontier = ordered[ordered[metric].gt(previous_best)].copy()
    frontier["observed_frontier"] = True

    maximum_budget = ordered[resource_column].max()
    if frontier[resource_column].iloc[-1] < maximum_budget:
        terminal = frontier.iloc[[-1]].copy()
        terminal[resource_column] = maximum_budget
        terminal["observed_frontier"] = False
        frontier = pd.concat([frontier, terminal], ignore_index=True)
    return frontier


def all_models_bwh(
    frame: pd.DataFrame, operations: pd.DataFrame, output_dir: Path
) -> None:
    """Plot all BWH candidates against operations and parameter count."""
    candidates = prepare(frame[frame["data"].eq("BWH")]).merge(
        operations,
        on=["model", "label"],
        how="left",
        validate="many_to_one",
    )
    missing = candidates[
        candidates[["model_macs", "model_params"]].isna().any(axis=1)
    ]["model"].unique()
    if len(missing):
        preview = ", ".join(map(str, missing[:5]))
        raise RuntimeError(
            f"Missing W&B complexity counts for {len(missing)} BWH models: {preview}"
        )
    candidates["total_operations"] = candidates["model_macs"]

    metrics = (
        ("roc_auc_boot_mean", "roc_auc_ci_low", "roc_auc_ci_high", "AUROC"),
        ("pr_auc_boot_mean", "pr_auc_ci_low", "pr_auc_ci_high", "AUPRC"),
    )
    figure_specs = (
        ("total_operations", "Model Operations", "Compute", "pareto_operations_bwh"),
        ("model_params", "Model Parameters", "Model Size", "pareto_params_bwh"),
    )

    def plot_resource_row(
        axes: np.ndarray,
        resource_column: str,
        xlabel: str,
        title_suffix: str,
    ) -> None:
        for panel_index, (metric, low, high, name) in enumerate(metrics):
            ax = axes[panel_index]
            for group_name, group in candidates.groupby("group"):
                color = COLORS[group_name]
                ax.scatter(
                    group[resource_column], group[metric],
                    s=10, color=color, alpha=0.18, edgecolors="none", zorder=1,
                )

                frontier = _budget_frontier(group, metric, resource_column)
                ax.fill_between(
                    frontier[resource_column], frontier[low], frontier[high],
                    color=color, alpha=0.15, edgecolor="none", zorder=3,
                )
                ax.plot(
                    frontier[resource_column], frontier[metric], "-o",
                    color=color, linewidth=1.8, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.5,
                    markevery=np.flatnonzero(frontier["observed_frontier"]).tolist(),
                    label=group_name.replace("+HR/HRV", "+ HR/HRV"), zorder=5,
                )

            ax.set_xscale("log")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.tick_params(axis="x", labelsize=10)
            ax.set_ylabel("")
            ax.set_title(
                f"{name} vs. {title_suffix}", fontweight="normal", pad=10
            )
            ax.set_ylim(0.5 if name == "AUROC" else 0.0, 1.02)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, which="both", linestyle="--", alpha=0.25)
            ax.legend(frameon=True, loc="lower right", fontsize=9)

    for resource_column, xlabel, title_suffix, output_stem in figure_specs:
        style()
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        plot_resource_row(axes, resource_column, xlabel, title_suffix)
        fig.tight_layout()
        save(fig, output_stem, output_dir)

    style()
    composite, composite_axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    plot_resource_row(
        composite_axes[0], "total_operations", "Model Operations", "Compute"
    )
    plot_resource_row(
        composite_axes[1], "model_params", "Model Parameters", "Model Size"
    )
    composite.tight_layout(h_pad=2.0)
    save(composite, "pareto_bwh", output_dir)


def six_panel(frame: pd.DataFrame, output_dir: Path, metric: str, low: str,
              high: str, name: str) -> None:
    style()
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
    for ax, dataset in zip(axes.flat, DATASETS):
        plot_axis(ax, frame[frame["data"].eq(dataset)], metric, low, high, name, confidence=True)
        ax.set_title(DATASET_NAMES[dataset], fontweight="bold", pad=12)
        ax.set_ylabel(name)
        ax.set_xlabel("Model FLOPs")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.subplots_adjust(top=0.93, bottom=0.08, hspace=0.6, wspace=0.4)
    save(fig, f"combined_pareto_{name.lower()}_6_panels", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = fetch_runs()
    simple_figures(frame, args.output_dir)
    bwh_combined(frame, args.output_dir)
    all_models_bwh(frame, fetch_training_operations(), args.output_dir)
    six_panel(frame, args.output_dir, "roc_auc_boot_mean", "roc_auc_ci_low", "roc_auc_ci_high", "AUROC")
    six_panel(frame, args.output_dir, "pr_auc_boot_mean", "pr_auc_ci_low", "pr_auc_ci_high", "AUPRC")


if __name__ == "__main__":
    main()
