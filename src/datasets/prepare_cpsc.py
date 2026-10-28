"""Discover CPSC windows from WFDB headers and rhythm annotations.

Assigns each fixed-duration window the rhythm covering the most samples.
Returns metadata with native sample boundaries for prepare_datasets; ECG samples
remain in the source files.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import csv
import json
import glob
import multiprocessing as mp
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

# Window-level rhythm labels:
#   - AFIB
#   - AFL
#   - OTHER = all other rhythm labels
WINDOW_VOCAB = ["OTHER", "AFIB", "AFL"]
WINDOW_TO_INDEX = {c: i for i, c in enumerate(WINDOW_VOCAB)}

# Record-level global labels from header comments
GLOBAL_VOCAB = ["N", "AFf", "AFp"]
GLOBAL_TO_INDEX = {c: i for i, c in enumerate(GLOBAL_VOCAB)}

SECONDS_LEN = 10
STRIDE_SECONDS = 10
N_LEADS = 1

BATCH_SIZE = 1000


# ============================================================
# HELPERS
# ============================================================


def _normalize_aux_note(note: Optional[str]) -> Optional[str]:
    """Map a WFDB rhythm note to AFIB/AFL/OTHER, or None for a non-rhythm note."""
    if note is None:
        return None

    note = str(note).strip().upper()
    if not note:
        return None

    if not note.startswith("("):
        return None

    token = note[1:]  # remove leading "("

    if token == "AFIB":
        return "AFIB"
    if token == "AFL":
        return "AFL"

    return "OTHER"


def _merge_same_label_intervals(
    intervals: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
    """Sort (start, end, label) sample intervals and merge adjacent same-label runs.

    Overlapping or touching intervals with the same label are joined; differently
    labeled intervals remain separate. End positions are exclusive.
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: (x[0], x[1], x[2]))
    merged: List[List[object]] = [[intervals[0][0], intervals[0][1], intervals[0][2]]]

    for s, e, label in intervals[1:]:
        ps, pe, plabel = merged[-1]

        if label == plabel and s <= pe:
            merged[-1][1] = max(pe, e)
        elif label == plabel and s == pe:
            merged[-1][1] = e
        else:
            merged.append([s, e, label])

    return [(int(s), int(e), str(label)) for s, e, label in merged]


def _class_overlaps(
    start: int,
    end: int,
    intervals: List[Tuple[int, int, str]],
) -> dict:
    """Return sample overlap totals for OTHER, AFIB, and AFL within [start, end).

    ``intervals`` contains (start, end, label) tuples in source sample indices.
    """
    overlaps = {"OTHER": 0, "AFIB": 0, "AFL": 0}

    for s, e, label in intervals:
        left = max(start, s)
        right = min(end, e)

        if right > left:
            overlaps[label] += right - left

    return overlaps


def _parse_global_label_from_header(base: str) -> str:
    """Read a WFDB record's global label from header comments.

    ``base`` omits the extension. Returns N, AFf (persistent AF), or AFp
    (paroxysmal AF); missing/unrecognized comments and read failures return N.
    """
    try:
        header = wfdb.rdheader(base)
    except Exception:
        return "N"

    comments = [str(c).strip().lower() for c in getattr(header, "comments", [])]

    if any("non atrial fibrillation" in c for c in comments):
        return "N"

    if any("persistent atrial fibrillation" in c for c in comments):
        return "AFf"

    if any("paroxysmal atrial fibrillation" in c for c in comments):
        return "AFp"

    return "N"


def _extract_rhythm_intervals(
    base: str,
    sig_len: int,
    global_label: str,
) -> List[Tuple[int, int, str]]:
    """Build record-wide rhythm intervals from WFDB annotations.

    Parameters
    ----------
    base : str
        Extensionless WFDB record path.
    sig_len : int
        Recording length in samples.
    global_label : str
        Recording label: N, AFf (persistent AF), or AFp (paroxysmal AF).

    Returns
    -------
    intervals : list of tuple
        ``(start, end, label)`` intervals in source samples, with an exclusive
        end and labels AFIB, AFL, or OTHER.

    Notes
    -----
    Missing or unusable annotations label the whole record AFIB for AFf,
    otherwise OTHER. The same rule labels any prefix before the first mark.
    """
    atr_path = base + ".atr"

    if not os.path.exists(atr_path):
        if global_label == "AFf":
            return [(0, int(sig_len), "AFIB")]
        return [(0, int(sig_len), "OTHER")]

    try:
        ann = wfdb.rdann(base, "atr")
    except Exception:
        if global_label == "AFf":
            return [(0, int(sig_len), "AFIB")]
        return [(0, int(sig_len), "OTHER")]

    marks: List[Tuple[int, str]] = []

    for sample, aux_note in zip(ann.sample, ann.aux_note):
        label = _normalize_aux_note(aux_note)

        if label is None:
            continue

        marks.append((int(sample), label))

    if not marks:
        if global_label == "AFf":
            return [(0, int(sig_len), "AFIB")]
        return [(0, int(sig_len), "OTHER")]

    intervals: List[Tuple[int, int, str]] = []

    # Prefix before first rhythm mark
    first_start = int(marks[0][0])

    if first_start > 0:
        prefix_label = "AFIB" if global_label == "AFf" else "OTHER"
        intervals.append((0, first_start, prefix_label))

    # Intervals between rhythm marks
    for i, (start, label) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else int(sig_len)

        if end > start:
            intervals.append((int(start), int(end), label))

    return _merge_same_label_intervals(intervals)


def _find_cpsc_record_bases(base_dir: str) -> List[Tuple[str, str]]:
    """Return (split_name, extensionless_record_path) pairs with .hea/.dat files.

    Accepts a training folder or a parent containing Training_set_I/II.
    """
    split_dirs = []

    t1 = os.path.join(base_dir, "Training_set_I")
    t2 = os.path.join(base_dir, "Training_set_II")

    if os.path.isdir(t1):
        split_dirs.append(("Training_set_I", t1))

    if os.path.isdir(t2):
        split_dirs.append(("Training_set_II", t2))

    if not split_dirs:
        split_dirs.append((os.path.basename(os.path.abspath(base_dir)), base_dir))

    rows = []

    for split_name, split_dir in split_dirs:
        for hea in sorted(glob.glob(os.path.join(split_dir, "*.hea"))):
            base = os.path.splitext(hea)[0]

            if os.path.exists(base + ".dat"):
                rows.append((split_name, base))

    return rows


# ============================================================
# DISCOVERY
# ============================================================


def _discover_windows_for_record(args):
    """Discover and label complete windows from one CPSC recording.

    Parameters
    ----------
    args : tuple
        ``(split_name, base, seconds_len, stride_seconds)``: training split,
        extensionless WFDB path, window duration, and stride in seconds.

    Returns
    -------
    rows : list of dict
        Window metadata with source sample offsets ``start`` and exclusive
        ``end``, ``fs`` in Hz, record IDs, and window/global labels.
        Empty for an unreadable header or a recording shorter than one window.

    Notes
    -----
    The window label has the greatest sample overlap. Ties favor AFIB/AFL over
    OTHER, then AFIB over AFL.
    """
    split_name, base, seconds_len, stride_seconds = args
    rows = []

    try:
        header = wfdb.rdheader(base)
    except Exception:
        return rows

    fs = int(round(float(header.fs)))
    sig_len = int(header.sig_len)
    record_id = os.path.basename(base)
    record_key = f"{split_name}/{record_id}"

    global_label = _parse_global_label_from_header(base)
    global_label_idx = GLOBAL_TO_INDEX[global_label]

    rhythm_intervals = _extract_rhythm_intervals(
        base=base,
        sig_len=sig_len,
        global_label=global_label,
    )

    win_len = fs * seconds_len
    stride_len = fs * stride_seconds

    if sig_len < win_len:
        return rows

    group_id = record_key

    for start in range(0, sig_len - win_len + 1, stride_len):
        end = start + win_len

        overlaps = _class_overlaps(start, end, rhythm_intervals)

        # Majority label inside the window.
        # Tie-break favors AFIB/AFL over OTHER.
        window_label = max(
            ["OTHER", "AFIB", "AFL"],
            key=lambda k: (overlaps[k], 1 if k != "OTHER" else 0),
        )

        label_idx = WINDOW_TO_INDEX[window_label]

        rows.append(
            {
                "base": base,
                "split_name": split_name,
                "record_id": record_id,
                "record_key": record_key,
                "group_id": group_id,
                "start": int(start),
                "end": int(end),
                "fs": int(fs),
                "sig_len": int(sig_len),
                "label_idx": int(label_idx),
                "window_label": window_label,
                "global_label_idx": int(global_label_idx),
                "global_label": global_label,
            }
        )

    return rows


def discover_cpsc_windows(
    base_dir: str,
    seconds_len: int = SECONDS_LEN,
    stride_seconds: int = STRIDE_SECONDS,
    num_workers: Optional[int] = None,
):
    """Discover CPSC windows in parallel and assign stable record group IDs.

    Parameters
    ----------
    base_dir : str
        Training directory or its parent containing Training_set_I/II.
    seconds_len : int, optional
        Window duration in seconds. Default is 10; incomplete tails are dropped.
    stride_seconds : int, optional
        Time between window starts in seconds. Default is 10.
    num_workers : int or None, optional
        Number of discovery processes. Default is None, using the CPU count.

    Returns
    -------
    rows : list of dict
        Window metadata in worker completion order, with source sample
        boundaries [start, end), fs in Hz, record IDs, and window/global labels.
        Windows from the same record share a stable ``group_id``.
    """
    record_bases = _find_cpsc_record_bases(base_dir)

    tasks = [
        (split_name, base, seconds_len, stride_seconds)
        for split_name, base in record_bases
    ]

    all_rows = []

    with mp.Pool(processes=num_workers or mp.cpu_count()) as pool:
        for rows in tqdm(
            pool.imap_unordered(_discover_windows_for_record, tasks),
            total=len(tasks),
            desc="Discovering CPSC windows",
        ):
            all_rows.extend(rows)

    # Stable integer group_id per record.
    group_map = {}
    next_gid = 0

    for row in sorted(
        all_rows, key=lambda x: (x["split_name"], x["record_id"], x["start"])
    ):
        key = row["group_id"]

        if key not in group_map:
            group_map[key] = next_gid
            next_gid += 1

        row["group_id"] = group_map[key]

    return all_rows
