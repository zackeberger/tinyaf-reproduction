"""Original physical samples and annotation windows, without preprocessing shards."""

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import wfdb
import h5py

COHORTS = ("SHDB", "IRIDIA", "CPSC")


def load_config(path="config.json"):
    """Read a JSON configuration file and return its dictionary unchanged.

    Dataset source and metadata paths remain relative to the working directory
    unless the JSON supplies absolute paths.
    """
    return json.loads(Path(path).read_text())


def patient_metadata_path(cohort, config):
    """Locate the published recording-to-patient mapping; CPSC has none."""
    if cohort == "CPSC":
        return None
    entry = config["datasets"][cohort]
    if "patient_metadata" in entry:
        path = Path(entry["patient_metadata"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    source = Path(entry["source"])
    filename = "AdditionalData.csv" if cohort == "SHDB" else "iridia-af-metadata-v1.0.1.csv"
    candidates = [source / filename]
    if cohort == "IRIDIA":
        candidates.append(source.parent.parent / filename)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {cohort} patient mapping; set patient_metadata in config")


def patient_ids(cohort, config, record_ids):
    """Map recordings to patient strings, rejecting missing or duplicate mappings."""
    path = patient_metadata_path(cohort, config)
    if path is None:
        return pd.Series(pd.NA, index=record_ids.index, dtype="string")
    record_col, patient_col = ("Data_ID", "Subject_ID") if cohort == "SHDB" else ("record_id", "patient_id")
    mapping = pd.read_csv(path, usecols=[record_col, patient_col], dtype="string")
    keys = record_ids.astype("string").str.strip()
    mapping[record_col] = mapping[record_col].str.strip()
    mapping[patient_col] = mapping[patient_col].str.strip()
    if mapping.isna().any().any() or mapping.eq("").any().any():
        raise ValueError(f"Invalid patient mapping in {path}")
    if cohort == "SHDB":
        keys = keys.str.zfill(3)
        mapping[record_col] = mapping[record_col].str.zfill(3)
    if mapping[record_col].duplicated().any():
        raise ValueError(f"Invalid or duplicate patient mapping in {path}")
    patients = keys.map(mapping.set_index(record_col)[patient_col]).astype("string")
    if patients.isna().any():
        missing = sorted(keys[patients.isna()].dropna().unique())
        raise ValueError(f"Missing {cohort} patient IDs for recordings: {missing[:10]}")
    return patients


def metadata(cohort, config):
    """Load one cohort's Parquet window manifests into a common evaluation table.

    Parameters
    ----------
    cohort : str
        Dataset name: ``"SHDB"``, ``"IRIDIA"``, or ``"CPSC"``.
    config : dict
        Dataset configuration. Reads Parquet files from
        ``config["datasets"][cohort]["metadata"]``.

    Returns
    -------
    windows : pandas.DataFrame
        Rows sorted by ``global_index``, with source sample boundaries,
        ``orig_fs`` in Hz, ``record_key``, ``patient_id`` (null for CPSC),
        and a binary AFIB/AFL label.
        ``eligible`` follows ``valid == 1`` when present, otherwise True.

    Raises
    ------
    FileNotFoundError
        If no Parquet manifests are found.
    ValueError
        If window IDs are duplicated.
    """
    entry = config["datasets"][cohort]
    files = sorted(Path(entry["metadata"]).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No annotation metadata under {entry['metadata']}")
    wanted = [
        "global_index",
        "record_id",
        "start_sample",
        "end_sample",
        "orig_fs",
    ]
    wanted += {
        "SHDB": ["AFIB", "AFL"],
        "IRIDIA": ["file_index", "AFIB"],
        "CPSC": ["record_key", "window_label", "valid"],
    }[cohort]
    frames = [pd.read_parquet(file, columns=wanted) for file in files]
    df = pd.concat(frames, ignore_index=True).sort_values("global_index")
    if df.global_index.duplicated().any():
        raise ValueError("duplicate window identity")
    if cohort == "CPSC":
        df["label"] = df.window_label.isin(["AFIB", "AFL"]).astype(np.uint8)
        df["record_key"] = df.record_key.astype(str)
    else:
        df["label"] = (
            (df.AFIB == 1) | (df.AFL == 1) if "AFL" in df else df.AFIB == 1
        ).astype(np.uint8)
        df["record_key"] = df.record_id.astype(str)
        if cohort == "IRIDIA":
            df["record_key"] += "/" + df.file_index.astype(str)
    df["eligible"] = df.get("valid", pd.Series(1, index=df.index)) == 1
    df["patient_id"] = patient_ids(cohort, config, df.record_id)
    return df.reset_index(drop=True)


def read_record(cohort, key, config):
    """Read original lead-0 samples without filtering or resampling.

    Parameters
    ----------
    cohort : str
        Dataset name: ``"SHDB"``, ``"IRIDIA"``, or ``"CPSC"``.
    key : str
        SHDB/CPSC WFDB record path relative to the configured source directory,
        or ``record_id/file_index`` for an IRIDIA HDF5 recording.
    config : dict
        Dataset configuration containing ``source`` and, for IRIDIA,
        ``native_fs`` in Hz under ``config["datasets"][cohort]``.

    Returns
    -------
    signal : numpy.ndarray, shape (n_samples,), dtype float64
        Original physical samples from lead 0.
    fs : int
        Sampling frequency in Hz, from the WFDB header or IRIDIA configuration.

    Raises
    ------
    ValueError
        If the rate is noninteger or an IRIDIA file lacks the ``ecg`` dataset.
    """
    entry = config["datasets"][cohort]
    source = Path(entry["source"])
    if cohort == "IRIDIA":
        rid, idx = key.split("/")
        p = source / rid / f"{rid}_ecg_{int(idx):02d}.h5"
        with h5py.File(p, "r") as f:
            if "ecg" not in f:
                raise ValueError(f"missing ecg dataset: {p}")
            data = np.asarray(f["ecg"][:, 0], dtype=np.float64)
        # Published IRIDIA acquisition rate; HDF5 has no sampling-rate attribute.
        fs = entry["native_fs"]
    else:
        rec = wfdb.rdrecord(str(source / key), channels=[0])
        data = np.asarray(rec.p_signal, dtype=np.float64).reshape(-1)
        fs = rec.fs
    if fs != int(fs):
        raise ValueError(f"noninteger native rate {fs}")
    return data, int(fs)


def metadata_provenance(config, cohorts=COHORTS):
    """Hash window manifests and patient mappings for evaluation provenance.

    Parameters
    ----------
    config : dict
        Dataset configuration containing each cohort's metadata directory.
    cohorts : iterable of str, optional
        Dataset names to include. Defaults to SHDB, IRIDIA, and CPSC.

    Returns
    -------
    hashes : dict
        ``{cohort: {input_path: sha256}}`` computed from file bytes.
    """
    cohorts = tuple(cohorts)
    result = {
        cohort: {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(
                Path(config["datasets"][cohort]["metadata"]).glob("*.parquet")
            )
        }
        for cohort in cohorts
    }
    for cohort in cohorts:
        path = patient_metadata_path(cohort, config)
        if path is not None:
            result[cohort][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
