"""Build annotation manifests; keep ECG in its original native-rate source files."""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .native import load_config
from .prepare_shdb import discover_shdb_segments
from .prepare_iridia import discover_iridia_segments, RHYTHM_VOCAB as IRIDIA_LABELS
from .prepare_cpsc import discover_cpsc_windows


def prepare(cohort, config, workers):
    """Discover one cohort's 10-second windows and write its evaluation manifests.

    Parameters
    ----------
    cohort : str
        Dataset name: ``"SHDB"``, ``"IRIDIA"``, or ``"CPSC"``.
    config : dict
        Contains source and output metadata directories under
        ``config["datasets"][cohort]``.
    workers : int or None
        Number of discovery processes. None uses the CPU count.

    Raises
    ------
    RuntimeError
        If no annotated windows are found.
    FileExistsError
        If the output directory already contains Parquet manifests.

    Notes
    -----
    Writes Parquet rows with stable global indices, recording IDs, source
    sample boundaries [start_sample, end_sample), orig_fs in Hz, and rhythm
    labels, plus provenance.json. Waveforms remain in their source files.
    """
    entry = config["datasets"][cohort]
    discover = {
        "SHDB": discover_shdb_segments,
        "IRIDIA": discover_iridia_segments,
        "CPSC": discover_cpsc_windows,
    }[cohort]
    rows = discover(entry["source"], num_workers=workers)
    if not rows:
        raise RuntimeError(f"No annotated windows discovered for {cohort}")
    d = pd.DataFrame(rows).rename(
        columns={"start": "start_sample", "end": "end_sample", "fs": "orig_fs"}
    )
    order = (
        ["record_id"]
        + (["file_index"] if cohort == "IRIDIA" else [])
        + ["start_sample", "end_sample"]
    )
    if cohort == "CPSC":
        order = ["record_key", "start_sample", "end_sample"]
    d = d.sort_values(order).reset_index(drop=True)
    d["global_index"] = np.arange(len(d), dtype=np.int64)
    if cohort == "SHDB":
        d["AFIB"] = (d.label == "AFIB").astype(np.uint8)
        d["AFL"] = (d.label == "AFL").astype(np.uint8)
    if cohort == "IRIDIA":
        d["AFIB"] = (d.label_idx == IRIDIA_LABELS.index("AFIB")).astype(np.uint8)
    if cohort == "CPSC":
        d["valid"] = 1
    cols = ["global_index", "record_id", "start_sample", "end_sample", "orig_fs"]
    cols += {
        "SHDB": ["AFIB", "AFL"],
        "IRIDIA": ["file_index", "AFIB"],
        "CPSC": ["record_key", "window_label", "valid"],
    }[cohort]
    out = Path(entry["metadata"])
    out.mkdir(parents=True, exist_ok=True)
    old = list(out.glob("*.parquet"))
    if old:
        raise FileExistsError(
            f"{out} already contains manifests; choose a fresh destination"
        )
    for start in range(0, len(d), 100000):
        d.iloc[start : start + 100000][cols].to_parquet(
            out / f"windows_{start//100000:04d}.parquet", index=False
        )
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "source": entry["source"],
                "cohort": cohort,
                "windows": len(d),
                "representation": "Original lead-0 physical samples, read as float64 at inference",
                "native_rates": d.orig_fs.value_counts().to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        cohort,
        len(d),
        "windows; native rates",
        d.orig_fs.value_counts().to_dict(),
        flush=True,
    )


def main():
    """Parse --config, --workers, and --cohorts, then prepare selected datasets."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--cohorts",
        nargs="+",
        choices=["SHDB", "IRIDIA", "CPSC"],
        default=["SHDB", "IRIDIA", "CPSC"],
    )
    a = ap.parse_args()
    c = load_config(a.config)
    for cohort in a.cohorts:
        prepare(cohort, c, a.workers)


if __name__ == "__main__":
    main()
