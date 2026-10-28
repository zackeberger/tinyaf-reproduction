"""Summarize selected predictions, comparing C with Python when both are present."""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(y, p):
    """Return average precision (``auprc``) and AUROC for binary labels and scores.

    Both values are None when ``y`` contains fewer than two distinct labels.
    """
    if len(np.unique(y)) < 2:
        return {"auprc": None, "auroc": None}
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
    }



def operating_metrics(y, p, threshold):
    """Confusion counts and rates on successful predictions; coverage is separate."""
    positive = p >= threshold
    y = y.astype(bool)
    tp, fn = int((positive & y).sum()), int((~positive & y).sum())
    fp, tn = int((positive & ~y).sum()), int((~positive & ~y).sum())
    ratio = lambda a, b: a / b if b else None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "precision": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn)}


def summarize(root):
    """Compute evaluation metrics and Python/C comparisons from saved predictions.

    Parameters
    ----------
    root : pathlib.Path
        Run directory containing ``manifest.json`` and per-recording Parquet
        files under ``SHDB``, ``IRIDIA``, and/or ``CPSC``. The manifest must
        contain the thresholds used for the saved run.

    Returns
    -------
    dict
        Results grouped by dataset and model, with eligible window counts,
        per-implementation metrics and statuses, and paired Python/C comparisons
        when both are present. Comparisons include logit bit mismatches,
        probability errors, classification agreement, and numerical gate results.

    Raises
    ------
    ValueError
        If the manifest lacks thresholds or window IDs repeat within a dataset.

    Notes
    -----
    Metrics include eligible windows with finite logits; paired comparisons
    use eligible windows with finite logits in both implementations. Status
    counts retain failures among eligible windows. Dataset window counts
    include all saved rows, including ineligible windows.

    Uses saved probabilities and classes when available, otherwise derives them
    from logits and the recorded thresholds. The CLI writes the returned
    dictionary to ``summary.json`` in the run directory.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    if "thresholds" not in manifest:
        raise ValueError(
            "Saved run has no threshold provenance; record its original "
            "thresholds before summarizing."
        )
    thresholds = manifest["thresholds"]
    result = {}
    for cohort in ["SHDB", "IRIDIA", "CPSC"]:
        paths = sorted((root / cohort).glob("*.parquet"))
        if not paths:
            continue
        df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        if df.global_index.duplicated().any():
            raise ValueError("duplicate output IDs")
        co = {
            "windows": len(df),
            "records": len(paths),
            "models": {},
        }
        if "python_status" in df:
            co["python_status"] = {
                str(k): int(v) for k, v in df.python_status.value_counts().items()
            }
        for r in manifest.get("models", thresholds):
            modes = [
                mode
                for mode in ["python", "reference", "optimized"]
                if mode + "_" + r in df
            ]
            if not modes:
                continue
            sub = df[df.eligible]
            y = sub.label.to_numpy()
            entry = {
                "windows": len(sub),
                "threshold": thresholds[r],
                "implementations": {},
                "comparisons": {},
            }
            co["models"][r] = entry
            probabilities = {}
            for mode in modes:
                logits = sub[mode + "_" + r].to_numpy()
                valid = np.isfinite(logits)
                probabilities[mode] = (
                    sub[mode + "_prob_" + r].to_numpy()
                    if mode + "_prob_" + r in sub
                    else expit(logits.astype(np.float64))
                )
                status_col = (
                    f"{mode}_status_{r}" if f"{mode}_status_{r}" in sub else "python_status"
                )
                entry["implementations"][mode] = {
                    "valid": int(valid.sum()),
                    "failed": int((~valid).sum()),
                    "coverage": float(valid.mean()) if len(valid) else None,
                    "operating_point": operating_metrics(y[valid], probabilities[mode][valid], thresholds[r]),
                    "metrics": metrics(y[valid], probabilities[mode][valid]),
                    "status_counts": {
                        str(k): int(v)
                        for k, v in sub[status_col].value_counts().items()
                    },
                }
            if "python" not in modes:
                continue
            py = sub["python_" + r].to_numpy()
            ok = np.isfinite(py)
            rp = probabilities["python"]
            # Preserve the existing summary keys for full comparison runs.
            entry.update(
                python_valid=int(ok.sum()), python=metrics(y[ok], rp[ok])
            )
            for mode in modes:
                if mode == "python":
                    continue
                c = sub[mode + "_" + r].to_numpy()
                valid = ok & np.isfinite(c)
                cp = probabilities[mode]
                delta = c[valid].astype(np.float64) - py[valid]
                base = metrics(y[valid], rp[valid])
                meas = metrics(y[valid], cp[valid])
                absolute = {
                    k: abs(meas[k] - base[k]) if meas[k] is not None else None
                    for k in base
                }
                mae = (
                    float(np.mean(np.abs(cp[valid] - rp[valid])))
                    if valid.any()
                    else None
                )
                agreement = (
                    float(
                        np.mean(
                            (
                                sub[mode + "_class_" + r].to_numpy()[valid]
                                if mode + "_class_" + r in sub
                                else cp[valid] >= thresholds[r]
                            )
                            == (rp[valid] >= thresholds[r])
                        )
                    )
                    if valid.any()
                    else None
                )
                bits = py[valid].view(np.uint32) == c[valid].view(np.uint32)
                entry["comparisons"][mode] = {
                    "paired_valid": int(valid.sum()),
                    "c_valid": int(np.isfinite(c).sum()),
                    "validity_mismatches": int(np.sum(np.isfinite(c) != ok)),
                    "status_counts": {
                        str(k): int(v)
                        for k, v in sub[mode + "_status_" + r]
                        .value_counts()
                        .items()
                    },
                    "bit_mismatches": int((~bits).sum()),
                    "max_abs_logit_delta": (
                        float(np.max(np.abs(delta))) if len(delta) else None
                    ),
                    "probability_mae": mae,
                    "classification_agreement": agreement,
                    "python_paired_metrics": base,
                    "c_metrics": meas,
                    "absolute_metric_delta": absolute,
                    "optimized_numeric_gates_pass": bool(
                        valid.any()
                        and mae < 0.001
                        and agreement > 0.999
                        and all(
                            v is not None and v < 0.001 for v in absolute.values()
                        )
                    ),
                }
        result[cohort] = co
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    args = p.parse_args()
    root = Path(args.root)
    out = summarize(root)
    (root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for cohort, c in out.items():
        print(cohort, c["windows"], "windows")
        for r, m in c["models"].items():
            for mode, v in m["implementations"].items():
                print(
                    r, mode, "valid", v["valid"], "metrics", v["metrics"],
                    "status", v["status_counts"],
                )
            for mode, v in m["comparisons"].items():
                print(
                    r,
                    mode,
                    "max",
                    v["max_abs_logit_delta"],
                    "MAE",
                    v["probability_mae"],
                    "agree",
                    v["classification_agreement"],
                    "metrics",
                    v["absolute_metric_delta"],
                    "validity",
                    v["validity_mismatches"],
                    "pass",
                    v["optimized_numeric_gates_pass"],
                )
