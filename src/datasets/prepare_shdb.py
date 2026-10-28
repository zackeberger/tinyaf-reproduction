"""Discover SHDB rhythm windows from WFDB headers and annotations.

Returns window metadata for prepare_datasets; ECG samples remain in source files.
Window boundaries use native sample indices with an exclusive end.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import glob
import json
import multiprocessing as mp
from typing import List, Optional

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
import csv

# ============================================================
# CONFIG
# ============================================================

RHYTHM_VOCAB = ["OTHER", "AFIB", "AFL", "AT", "SVT"]
RHYTHM_TO_INDEX = {c: i for i, c in enumerate(RHYTHM_VOCAB)}

AUX_TO_LABEL = {
    "(N": "OTHER",
    "(AFIB": "AFIB",
    "(AFL": "AFL",
    "(AT": "AT",
    "(PAT": "SVT",
    "(NOD": "SVT",
}

INCLUDE_OTHER = True

SECONDS_LEN = 10
N_LEADS = 1

BATCH_SIZE = 1000


# ============================================================
# DISCOVERY
# ============================================================


def _normalize_aux_note(note: Optional[str]) -> Optional[str]:
    """Map a recognized WFDB rhythm prefix to RHYTHM_VOCAB, or return None."""
    if note is None:
        return None

    note = note.strip()
    if not note:
        return None

    for prefix, label in AUX_TO_LABEL.items():
        if note.startswith(prefix):
            return label

    return None


def _find_record_bases(base_dir: str) -> List[str]:
    """Return sorted extensionless paths with both .hea and .dat in base_dir."""
    bases = []

    for hea in sorted(glob.glob(os.path.join(base_dir, "*.hea"))):
        base = os.path.splitext(hea)[0]
        if os.path.exists(base + ".dat"):
            bases.append(base)

    return bases


def _discover_intervals_for_record(args):
    """Split one SHDB record's recognized rhythm intervals into complete windows.

    Parameters
    ----------
    args : tuple
        ``(record_base, seconds_len, include_other)``: extensionless WFDB path,
        window duration in seconds, and whether to retain OTHER rhythm marks.

    Returns
    -------
    rows : list of dict
        Window metadata with record ID, source sample offsets ``start`` and
        exclusive ``end``, ``fs`` in Hz, label/index, and interval key.
        Empty for missing/unreadable headers or annotations, or no full windows.
    """
    base, seconds_len, include_other = args
    rows = []

    if not os.path.exists(base + ".atr"):
        return rows

    try:
        header = wfdb.rdheader(base)
        ann = wfdb.rdann(base, "atr")
    except Exception:
        return rows

    fs = int(round(float(header.fs)))
    sig_len = int(header.sig_len)
    samples_per_clip = fs * seconds_len
    record_id = os.path.basename(base)

    marks = []
    for sample, note in zip(ann.sample, ann.aux_note):
        label = _normalize_aux_note(note)

        if label is None:
            continue

        if not include_other and label == "OTHER":
            continue

        marks.append((int(sample), label))

    if len(marks) == 0:
        return rows

    interval_counter = 0

    for idx, (start, label) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else sig_len

        if end <= start:
            continue

        interval_len = end - start

        if interval_len < samples_per_clip:
            continue

        n_chunks = interval_len // samples_per_clip
        label_idx = RHYTHM_TO_INDEX[label]
        interval_key = f"{record_id}__{interval_counter}"

        for k in range(n_chunks):
            chunk_start = start + k * samples_per_clip

            rows.append(
                {
                    "base": base,
                    "record_id": record_id,
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


def discover_shdb_segments(
    base_dir: str,
    seconds_len: int = SECONDS_LEN,
    num_workers: Optional[int] = None,
):
    """Discover SHDB rhythm windows in parallel.

    Parameters
    ----------
    base_dir : str
        Directory containing WFDB recording and annotation files.
    seconds_len : int, optional
        Window duration in seconds. Default is 10; incomplete tails are dropped.
    num_workers : int or None, optional
        Number of discovery processes. Default is None, using the CPU count.

    Returns
    -------
    rows : list of dict
        Window metadata in worker completion order, with source sample
        boundaries [start, end), fs in Hz, record IDs, and rhythm labels.
        Windows from the same rhythm interval share a ``group_id``.
    """
    record_bases = _find_record_bases(base_dir)
    tasks = [(b, seconds_len, INCLUDE_OTHER) for b in record_bases]

    all_rows = []

    with mp.Pool(processes=num_workers or mp.cpu_count()) as pool:
        for rows in tqdm(
            pool.imap_unordered(_discover_intervals_for_record, tasks),
            total=len(tasks),
            desc="Discovering SHDB intervals",
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
