#!/usr/bin/env python3
"""Plot true versus TinyAF-S predicted AF/AFL burden for each patient.

The saved classes use the operating threshold recorded in manifest.json.
SHDB truth includes AF/AFL; IRIDIA truth uses its available AF annotations.

By default, only successfully scored eligible windows contribute to either
burden: failed windows are excluded from both numerators and the shared
denominator. Use `--denominator recorded` to include failed windows in total
time and true AF/AFL time, with no predicted AF/AFL contribution.

Writes patient_burden.png, patient_burden.pdf, and the plotted patient-level
values in patient_burden.csv to fig/outputs. No inference is performed.

    python fig/scripts/patient_burden.py
    python fig/scripts/patient_burden.py --denominator recorded
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PREDICTION_DIR = ROOT / "logs/evaluation-tinyaf-s-patients"
OUTPUT_DIR = ROOT / "fig/outputs"
COHORTS = ("SHDB", "IRIDIA")
DISPLAY_NAMES = {"SHDB": "SHDB-AF", "IRIDIA": "IRIDIA-AF"}
MODEL = "TINYAF-S"
STATUS = f"python_status_{MODEL}"
PROBABILITY = f"python_prob_{MODEL}"
CLASS = f"python_class_{MODEL}"
WINDOW_SECONDS = 10
TINYAF_COLOR = "#1583F2"
COLUMNS = ["global_index", "patient_id", "label", "eligible", STATUS,
           PROBABILITY, CLASS]


def patient_times(frame: pd.DataFrame, threshold: float,
                  denominator: str) -> pd.DataFrame:
    """Sum durations by patient; keep unscored time explicit for coverage."""
    if denominator not in ("recorded", "scored"):
        raise ValueError("denominator must be recorded or scored")
    if frame.patient_id.isna().any() or frame.patient_id.astype(str).str.strip().eq("").any():
        raise ValueError("Predictions contain missing patient IDs")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("True labels must be binary AF/AFL indicators")
    if not frame.eligible.isin([True, False]).all() or frame[STATUS].isna().any():
        raise ValueError("Predictions contain invalid eligibility or status values")
    frame = frame.loc[frame.eligible.astype(bool)].copy()
    scored = frame[STATUS].eq(0)
    if not frame.loc[scored, PROBABILITY].between(0, 1).all():
        raise ValueError("Successful predictions must have finite probabilities in [0, 1]")
    expected_class = frame.loc[scored, PROBABILITY].ge(threshold).astype(int)
    if not np.array_equal(frame.loc[scored, CLASS].to_numpy(), expected_class.to_numpy()):
        raise ValueError("Saved classes disagree with the threshold in manifest.json")
    included = scored if denominator == "scored" else pd.Series(True, index=frame.index)
    times = pd.DataFrame({
        "patient_id": frame.patient_id.astype("string"),
        "recorded_seconds": WINDOW_SECONDS,
        "scored_seconds": scored.astype(int) * WINDOW_SECONDS,
        "analyzed_seconds": included.astype(int) * WINDOW_SECONDS,
        "true_af_seconds": (included & frame.label.eq(1)).astype(int) * WINDOW_SECONDS,
        "predicted_af_seconds": (scored & frame[CLASS].eq(1)).astype(int) * WINDOW_SECONDS,
    })
    return times.groupby("patient_id", sort=True).sum()


def load_burdens(prediction_dir: Path, denominator: str = "scored"):
    """Read prediction files incrementally, merging patients across recordings."""
    manifest = json.loads((prediction_dir / "manifest.json").read_text())
    if MODEL not in manifest.get("models", []) or "python" not in manifest.get("implementations", []):
        raise ValueError("The run must contain TINYAF-S Python predictions")
    if not set(COHORTS).issubset(manifest.get("cohorts", [])):
        raise ValueError("The run must include SHDB and IRIDIA")
    if manifest.get("max_per_record") != 0:
        raise ValueError("Patient burden requires a full run, not sampled windows")
    threshold = float(manifest["thresholds"][MODEL])
    if not 0 <= threshold <= 1:
        raise ValueError("The saved operating threshold must be in [0, 1]")
    validation_path = prediction_dir / "validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    results = {}
    for cohort in COHORTS:
        files = sorted((prediction_dir / cohort).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No prediction files under {prediction_dir / cohort}")
        pieces, identities = [], []
        for path in files:
            frame = pd.read_parquet(path, columns=COLUMNS)
            if frame.global_index.isna().any():
                raise ValueError(f"Missing window identity in {path}")
            identities.append(frame.global_index.to_numpy())
            pieces.append(patient_times(frame, threshold, denominator))
        identities = np.concatenate(identities)
        if len(np.unique(identities)) != len(identities):
            raise ValueError(f"Duplicate {cohort} prediction windows")
        if cohort in validation and len(identities) != validation[cohort]["windows"]:
            raise ValueError(f"{cohort} prediction count differs from its validation report")
        table = pd.concat(pieces).groupby(level="patient_id", sort=True).sum()
        no_time = table.analyzed_seconds.eq(0)
        if no_time.any():
            print(f"{cohort}: excluding {int(no_time.sum())} patients with no analyzed time")
        table = table.loc[~no_time].copy()
        if table.empty:
            raise ValueError(f"No analyzable patients in {cohort}")
        table["true_burden_pct"] = 100 * table.true_af_seconds / table.analyzed_seconds
        table["predicted_burden_pct"] = 100 * table.predicted_af_seconds / table.analyzed_seconds
        table["scored_coverage_pct"] = 100 * table.scored_seconds / table.recorded_seconds
        results[cohort] = table
    return results, threshold


def error_summary(table: pd.DataFrame) -> dict:
    """Patient-weighted absolute burden errors in percentage points."""
    error = np.abs(table.predicted_burden_pct - table.true_burden_pct)
    return {"mae": float(error.mean()), "median": float(error.median()),
            "p25": float(error.quantile(0.25)), "p75": float(error.quantile(0.75))}


def make_figure(results: dict):
    """Draw one point per patient, with predicted burden across and truth up."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), sharex=True, sharey=True,
                             layout="constrained")
    fig.get_layout_engine().set(wspace=0.07)
    for axis, cohort in zip(axes, COHORTS):
        table = results[cohort]
        axis.plot([0, 100], [0, 100], color="black", linestyle="--",
                  linewidth=1.5, zorder=1)
        axis.scatter(table.predicted_burden_pct, table.true_burden_pct,
                     s=42, color=TINYAF_COLOR, alpha=0.8,
                     edgecolors="white", linewidths=0.45, zorder=3)
        axis.set_title(DISPLAY_NAMES[cohort], fontsize=20, pad=14)
        axis.text(0.96, 0.045, f"MAE {error_summary(table)['mae']:.1f}%",
                  transform=axis.transAxes, ha="right", va="bottom", fontsize=16)
        axis.set(xlim=(-4, 104), ylim=(-4, 104), xticks=[0, 25, 50, 75, 100],
                 yticks=[0, 25, 50, 75, 100])
        axis.set_aspect("equal", adjustable="box")
        axis.set_axisbelow(True)
        axis.grid(True, linestyle="--", color="#C6C6C6", alpha=0.55, linewidth=0.7)
        axis.tick_params(axis="both", labelsize=16, pad=7)
        axis.spines[["top", "right"]].set_visible(False)
    xlabel = fig.supxlabel("Predicted burden (%)", fontsize=20)
    ylabel = fig.supylabel("True burden (%)", fontsize=20)
    # Match the physical distance from each panel edge to its shared label.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel = axes[0].get_window_extent(renderer)
    y_gap = panel.x0 - ylabel.get_window_extent(renderer).x1
    x_gap = panel.y0 - xlabel.get_window_extent(renderer).y1
    fig.set_layout_engine("none")
    xlabel.set_y(xlabel.get_position()[1] - (y_gap - x_gap) / fig.bbox.height)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--denominator", choices=("scored", "recorded"), default="scored",
                        help="Successfully scored time for both axes (default), or all eligible saved time")
    args = parser.parse_args()
    results, threshold = load_burdens(args.predictions_dir, args.denominator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure(results)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"patient_burden.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    values = pd.concat(results, names=["cohort", "patient_id"]).reset_index()
    values["denominator"] = args.denominator
    values["threshold"] = threshold
    values.to_csv(args.output_dir / "patient_burden.csv", index=False)
    print(f"TinyAF (TinyAF-S), Python; threshold {threshold:.9f}; denominator={args.denominator}")
    print("Time refers to eligible complete 10-second windows in the saved predictions.")
    for cohort, table in results.items():
        stats = error_summary(table)
        failed = int((table.recorded_seconds.sum() - table.scored_seconds.sum()) / WINDOW_SECONDS)
        print(f"{DISPLAY_NAMES[cohort]}: {len(table)} patients; {failed} failed windows; "
              f"MAE {stats['mae']:.2f}, median {stats['median']:.2f}, "
              f"IQR [{stats['p25']:.2f}, {stats['p75']:.2f}] percentage points")
    print(f"Wrote {args.output_dir / 'patient_burden'}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
