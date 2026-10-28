"""Discover IRIDIA AFIB/OTHER windows from CSV labels and HDF5 recording lengths.

Uses the released 200 Hz rate and, by default, excludes 6,000 calibration samples
at the start of every file. Returns metadata with native sample boundaries for
prepare_datasets; discovery does not load or rewrite ECG waveforms.
"""

from __future__ import annotations

import os

# ------------------------------------------------------------------
# IMPORTANT: set thread limits BEFORE importing numpy/scipy stack
# ------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import glob
import json
import h5py
import socket
import multiprocessing as mp
from typing import List, Optional

import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

RHYTHM_VOCAB = ["OTHER", "AFIB"]
RHYTHM_TO_INDEX = {c: i for i, c in enumerate(RHYTHM_VOCAB)}

INCLUDE_OTHER = True

ORIGINAL_FS = 200
SECONDS_LEN = 10
N_LEADS = 1

BATCH_SIZE = 1000

DROP_CALIBRATION = True
CALIBRATION_SAMPLES = 6000
DROP_CALIBRATION_MODE = "each_file"  # "each_file" or "first_file_only"


# ============================================================
# HELPERS
# ============================================================


def _find_record_dirs(base_dir: str) -> List[str]:
    """Return sorted record_* directories containing *_ecg_*.h5 below base_dir."""
    record_dirs = []

    pattern = os.path.join(base_dir, "**", "record_*")

    for p in sorted(glob.glob(pattern, recursive=True)):
        if os.path.isdir(p) and glob.glob(os.path.join(p, "*_ecg_*.h5")):
            record_dirs.append(p)

    return record_dirs


def _h5_first_key(f: h5py.File) -> str:
    """Return the first top-level HDF5 key; raise RuntimeError for an empty file."""
    keys = list(f.keys())

    if len(keys) == 0:
        raise RuntimeError("HDF5 file has no datasets")

    return keys[0]


def _read_h5_len(h5_path: str) -> int:
    """Return the first HDF5 dataset's sample count without loading its values."""
    with h5py.File(h5_path, "r") as f:
        key = _h5_first_key(f)
        return int(f[key].shape[0])


def _merge_intervals(
    intervals: List[tuple[int, int]],
    lo: int,
    hi: int,
) -> List[tuple[int, int]]:
    """Clip sample intervals to [lo, hi), discard empty ones, and merge overlaps.

    Returns sorted (start, end) pairs; touching intervals are also joined.
    """
    cleaned = []

    for s, e in intervals:
        s = max(int(s), lo)
        e = min(int(e), hi)

        if e > s:
            cleaned.append((s, e))

    if not cleaned:
        return []

    cleaned.sort()
    merged = [cleaned[0]]

    for s, e in cleaned[1:]:
        ps, pe = merged[-1]

        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    return merged


def _complement_intervals(
    intervals: List[tuple[int, int]],
    lo: int,
    hi: int,
) -> List[tuple[int, int]]:
    """Return uncovered sample intervals within [lo, hi).

    Expects sorted, merged input intervals already clipped to these bounds.
    Output pairs use an exclusive end; empty bounds produce an empty list.
    """
    if hi <= lo:
        return []

    if not intervals:
        return [(lo, hi)]

    out = []
    cur = lo

    for s, e in intervals:
        if s > cur:
            out.append((cur, s))

        cur = max(cur, e)

    if cur < hi:
        out.append((cur, hi))

    return out


def _calibration_offset(file_index: int) -> int:
    """Return samples to skip for a zero-based ECG file index.

    Uses DROP_CALIBRATION and DROP_CALIBRATION_MODE; the default skips 6,000
    samples (30 seconds at 200 Hz) in every file. Rejects an unknown active mode.
    """
    if not DROP_CALIBRATION:
        return 0

    if DROP_CALIBRATION_MODE == "each_file":
        return CALIBRATION_SAMPLES

    if DROP_CALIBRATION_MODE == "first_file_only":
        return CALIBRATION_SAMPLES if file_index == 0 else 0

    raise ValueError(f"Invalid DROP_CALIBRATION_MODE: {DROP_CALIBRATION_MODE}")


def _read_labels_csv(label_path: str) -> pd.DataFrame:
    """Read AFIB interval labels and require start/end file and QRS index columns.

    Returns the original DataFrame; raises ValueError if required columns are
    missing. Discovery uses the QRS indices as sample positions within each file.
    """
    df = pd.read_csv(label_path)

    required = {
        "start_file_index",
        "start_qrs_index",
        "end_file_index",
        "end_qrs_index",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing label columns in {label_path}: {sorted(missing)}")

    return df


# ============================================================
# DISCOVERY
# ============================================================


def _discover_intervals_for_record(args):
    """Discover IRIDIA windows within individual files and rhythm intervals.

    Splits CSV AFIB intervals across ECG files and excludes calibration samples.

    Parameters
    ----------
    args : tuple
        ``(record_dir, seconds_len, include_other)``: recording directory,
        window duration in seconds, and whether to include non-AFIB intervals.

    Returns
    -------
    rows : list of dict
        Complete windows with file/record IDs, source sample offsets ``start``
        and exclusive ``end``, ``fs`` in Hz, label/index, and interval key.
        Empty if labels or recording lengths are missing or unreadable.

    Notes
    -----
    Invalid label rows are skipped. When ``include_other`` is true, samples
    outside AFIB intervals and the calibration region are labeled OTHER.
    """
    record_dir, seconds_len, include_other = args
    rows = []

    record_id = os.path.basename(record_dir)
    ecg_files = sorted(glob.glob(os.path.join(record_dir, "*_ecg_*.h5")))
    label_files = sorted(glob.glob(os.path.join(record_dir, "*ecg_labels.csv")))

    if len(ecg_files) == 0 or len(label_files) != 1:
        return rows

    try:
        labels_df = _read_labels_csv(label_files[0])
    except Exception:
        return rows

    fs = ORIGINAL_FS
    samples_per_clip = fs * seconds_len

    try:
        file_lengths = [_read_h5_len(p) for p in ecg_files]
    except Exception:
        return rows

    n_files = len(ecg_files)
    af_by_file = {i: [] for i in range(n_files)}

    for _, row in labels_df.iterrows():
        try:
            sfi = int(row["start_file_index"])
            sqi = int(row["start_qrs_index"])
            efi = int(row["end_file_index"])
            eqi = int(row["end_qrs_index"])
        except Exception:
            continue

        if not (0 <= sfi < n_files and 0 <= efi < n_files):
            continue

        if efi < sfi:
            continue

        if sfi == efi:
            af_by_file[sfi].append((sqi, eqi))
        else:
            af_by_file[sfi].append((sqi, file_lengths[sfi]))

            for mid in range(sfi + 1, efi):
                af_by_file[mid].append((0, file_lengths[mid]))

            af_by_file[efi].append((0, eqi))

    interval_counter = 0

    for file_index, (file_path, file_len) in enumerate(zip(ecg_files, file_lengths)):
        lo = _calibration_offset(file_index)
        hi = file_len

        if hi - lo < samples_per_clip:
            continue

        af_intervals = _merge_intervals(
            af_by_file[file_index],
            lo=lo,
            hi=hi,
        )

        labeled_intervals = [(s, e, "AFIB") for s, e in af_intervals]

        if include_other:
            other_intervals = _complement_intervals(
                af_intervals,
                lo=lo,
                hi=hi,
            )

            labeled_intervals.extend((s, e, "OTHER") for s, e in other_intervals)

        labeled_intervals.sort(key=lambda x: (x[0], x[1]))

        for s, e, label in labeled_intervals:
            if e - s < samples_per_clip:
                continue

            n_chunks = (e - s) // samples_per_clip
            label_idx = RHYTHM_TO_INDEX[label]

            interval_key = (
                f"{record_id}__file{file_index:02d}" f"__{label}__{interval_counter}"
            )

            for k in range(n_chunks):
                chunk_start = s + k * samples_per_clip

                rows.append(
                    {
                        "record_dir": record_dir,
                        "record_id": record_id,
                        "file_path": file_path,
                        "file_index": int(file_index),
                        "start": int(chunk_start),
                        "end": int(chunk_start + samples_per_clip),
                        "fs": fs,
                        "label": label,
                        "label_idx": label_idx,
                        "interval_key": interval_key,
                    }
                )

            interval_counter += 1

    return rows


def discover_iridia_segments(
    base_dir: str,
    seconds_len: int = SECONDS_LEN,
    num_workers: Optional[int] = None,
):
    """Discover IRIDIA rhythm windows in parallel.

    Parameters
    ----------
    base_dir : str
        Directory containing record_* folders with HDF5 ECG files and CSV labels.
    seconds_len : int, optional
        Window duration in seconds. Default is 10; incomplete tails are dropped.
    num_workers : int or None, optional
        Number of discovery processes. Default is None, using the CPU count.

    Returns
    -------
    rows : list of dict
        Window metadata in worker completion order, with file/record IDs,
        source sample boundaries [start, end), fs in Hz, and rhythm labels.
        Each window stays within one file and rhythm interval; windows from
        the same interval share a ``group_id``.

    Raises
    ------
    RuntimeError
        If no record directories containing ECG files are found.
    """
    record_dirs = _find_record_dirs(base_dir)

    if len(record_dirs) == 0:
        raise RuntimeError(f"No IRIDIA record_* directories found under {base_dir}")

    tasks = [(rd, seconds_len, INCLUDE_OTHER) for rd in record_dirs]

    all_rows = []

    with mp.Pool(processes=num_workers or mp.cpu_count()) as pool:
        for rows in tqdm(
            pool.imap_unordered(_discover_intervals_for_record, tasks),
            total=len(tasks),
            desc="Discovering IRIDIA intervals",
        ):
            all_rows.extend(rows)

    group_map = {}
    next_gid = 0

    for row in all_rows:
        key = row["interval_key"]

        if key not in group_map:
            group_map[key] = next_gid
            next_gid += 1

        row["group_id"] = group_map[key]

    return all_rows
