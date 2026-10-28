"""Evaluate selected models and implementations on raw native windows.

Preprocessing is computed once per raw window for the selected Python models; no
preprocessed ECG/beat shards are read or written. C receives the same raw window.
"""

import os

for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tinyaf-mpl")
import argparse, ctypes as ct, json, time, hashlib, warnings, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from python_model.runtime import (
    configure_reference, load_model, THRESHOLDS, MODELS, TINYAF_MODELS,
    MODEL_NAMES, predict_prepared, predict_batch,
    evaluation_threshold,
)
from python_model.pipeline import prepare_window
from scipy.special import expit
from src.bindings import Predictor
from src.datasets.native import (
    COHORTS,
    load_config,
    metadata,
    metadata_provenance,
    read_record,
)

IMPLEMENTATIONS = ("python", "reference", "optimized")


def initialize(config, models=TINYAF_MODELS, implementations=IMPLEMENTATIONS, nn_devices=("cpu",), nn_batch_size=1):
    """Load selected Python models on the worker's device and enable batching."""
    global CONFIG, NETS, C, CUTS, NN_BATCH_SIZE, BATCH_MODELS
    CONFIG = config
    NN_BATCH_SIZE = nn_batch_size
    configure_reference()
    worker = (mp.current_process()._identity or (1,))[0] - 1
    device = nn_devices[worker % len(nn_devices)]
    NETS = {
        r: load_model(r, device) for r in models
    } if "python" in implementations else {}
    BATCH_MODELS = NETS.copy()
    C = {
        (mode, r): Predictor(MODELS[r].rung, mode)
        for mode in implementations if mode != "python"
        for r in models if mode in MODELS[r].implementations
    }
    CUTS = {r: evaluation_threshold(r) for r in models}
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="neurokit2")


def score_window(raw, fs, models):
    """Prepare one raw window, recording failures before deferred batch scoring."""
    window = prepare_window(raw, fs, {m.input_kind for m in models.values()})
    immediate = {r: net for r, net in models.items() if r not in BATCH_MODELS or window.status}
    return window, predict_prepared(window, immediate)


def save_python(result, predictions):
    for r, pred in predictions.items():
        result[f"python_{r}"] = pred.logit
        result[f"python_status_{r}"] = pred.status
        result[f"python_error_{r}"] = pred.error
        if pred.status == 0:
            probability = float(expit(pred.logit))
            result[f"python_prob_{r}"] = probability
            result[f"python_class_{r}"] = int(probability >= CUTS[r])


def job(task):
    """Read a recording and process each original window live, preserving all rows.

    Each model has its own status/error columns. No saved ECG or feature shards
    are read. Existing prediction files are reusable only under the same manifest.
    """
    cohort, key, rows, output = task
    output = Path(output)
    if output.is_file():
        return len(rows), "cached", str(output)
    read_error = ""
    try:
        sig, fs = read_record(cohort, key, CONFIG)
    except Exception as exc:
        sig, fs = None, int(rows.orig_fs.iloc[0])
        read_error = type(exc).__name__ + ": " + str(exc)[:160]
    values = []
    batch_models = BATCH_MODELS
    pending = []

    def flush():
        if not pending:
            return
        predictions = predict_batch([w for w, _ in pending], batch_models)
        for (_, result), scores in zip(pending, predictions):
            save_python(result, scores)
        pending.clear()

    for row in rows.itertuples(index=False):
        result = {
            "global_index": row.global_index, "label": row.label,
            "eligible": row.eligible, "fs": fs,
            "start_sample": row.start_sample, "source_key": key,
            "patient_id": row.patient_id,
        }
        pairs = [("python", r) for r in NETS] + list(C)
        for mode, r in pairs:
            result.update({f"{mode}_{r}": np.nan, f"{mode}_status_{r}": 0,
                           f"{mode}_error_{r}": "", f"{mode}_prob_{r}": np.nan,
                           f"{mode}_class_{r}": -1})
        a, b = int(row.start_sample), int(row.end_sample)
        status, error = 0, ""
        if read_error:
            status, error = -11, read_error
        elif fs != row.orig_fs or a < 0 or b > len(sig) or b - a != 10 * fs:
            status, error = -2, "invalid native window or rate"
        elif not row.eligible:
            status, error = -10, "ineligible window"
        if status:
            for mode, r in pairs:
                result[f"{mode}_status_{r}"] = status
                result[f"{mode}_error_{r}"] = error
            values.append(result)
            continue
        raw = np.ascontiguousarray(sig[a:b], dtype=np.float64)
        if NETS:
            window, predictions = score_window(raw, fs, NETS)
            save_python(result, predictions)
            if batch_models and not window.status:
                pending.append((window, result))
                if len(pending) >= NN_BATCH_SIZE:
                    flush()
        for (mode, r), c in C.items():
            status, out = c.native(raw, fs)
            result[f"{mode}_status_{r}"] = status
            result[f"{mode}_{r}"] = out.logit
            result[f"{mode}_prob_{r}"] = out.probability
            result[f"{mode}_class_{r}"] = out.prediction
        values.append(result)
    flush()
    frame = pd.DataFrame(values)
    frame["patient_id"] = frame["patient_id"].astype("string")
    for col in ["python_" + r for r in NETS] + [m + "_" + r for m, r in C]:
        frame[col] = frame[col].astype(np.float32)
    tmp = output.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(output)
    return len(frame), "computed", str(output)


def named_selection(value):
    """Uppercase a model or dataset name, preserving the special value ``all``."""
    return "all" if value.lower() == "all" else value.upper()


def expand_selection(values, choices):
    """Expand ``all`` and deduplicate selections in the order of ``choices``."""
    # Canonical ordering makes equivalent selections resumable and removes duplicates.
    return [v for v in choices if "all" in values or v in values]


def main():
    """Parse CLI options and evaluate the selected recordings in worker processes.

    Write ``manifest.json`` and per-recording prediction files under ``--output``.
    Matching runs reuse completed recordings. ``--max-per-record`` optionally
    selects evenly spaced windows from each recording; zero selects all windows.

    Raises
    ------
    RuntimeError
        If an existing output manifest differs from the current inputs,
        selections, thresholds, reference settings, or recorded code/assets.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-per-record", type=int, default=0)
    ap.add_argument(
        "--cohorts",
        nargs="+", type=named_selection, choices=[*COHORTS, "all"], default=["all"],
        help="Datasets to evaluate (default: all; names are case-insensitive).",
    )
    ap.add_argument(
        "--models",
        nargs="+", type=named_selection,
        choices=[*MODEL_NAMES, "all"], default=list(TINYAF_MODELS),
        help="TINYAF-S, TINYAF-M, TINYAF-L, LARGENN, SMALLNN, HRHRV, or all (default: TINYAF-S TINYAF-M TINYAF-L).",
    )
    ap.add_argument(
        "--implementation", dest="implementations", nargs="+",
        type=str.lower, choices=[*IMPLEMENTATIONS, "all"], default=["all"],
        help="Implementations to run (default: all).",
    )
    ap.add_argument("--nn-device", nargs="+", default=["cpu"],
                    help="Devices for all Python models, e.g. cpu or cuda:0 cuda:1. Workers rotate across devices.")
    ap.add_argument("--nn-batch-size", type=int, default=1,
                    help="Forward batch size for all Python models; raw preprocessing remains per window (default: 1).")
    args = ap.parse_args()
    if args.workers < 1 or args.max_per_record < 0 or args.nn_batch_size < 1:
        ap.error("--workers and --nn-batch-size must be positive; --max-per-record must be nonnegative")
    args.cohorts = expand_selection(args.cohorts, COHORTS)
    args.models = expand_selection(args.models, MODEL_NAMES)
    args.implementations = expand_selection(args.implementations, IMPLEMENTATIONS)
    for r in args.models:
        if not set(args.implementations).intersection(MODELS[r].implementations):
            ap.error(f"{r} supports Python only; include --implementation python")
    python_selected = "python" in args.implementations
    devices = args.nn_device if python_selected else ["cpu"]
    configure_reference()
    gpu_info = {}
    for value in devices:
        try:
            device = torch.device(value)
            if device.type not in ("cpu", "cuda"):
                raise ValueError("only cpu and cuda devices are supported")
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise ValueError("CUDA is unavailable; no automatic CPU fallback")
                gpu_info[value] = torch.cuda.get_device_name(device)
        except (ValueError, RuntimeError, AssertionError) as exc:
            ap.error(f"Invalid --nn-device {value}: {exc}")
    config = load_config(args.config)
    missing = set(args.cohorts) - config["datasets"].keys()
    if missing:
        ap.error("Selected datasets missing from config: " + ", ".join(sorted(missing)))
    config = {**config, "datasets": {c: config["datasets"][c] for c in args.cohorts}}
    output = Path(args.output)
    hashfile = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    c_modes = [m for m in args.implementations if m != "python"]
    libraries = {}
    for mode in c_modes:
        for r in args.models:
            if mode not in MODELS[r].implementations:
                continue
            rung = MODELS[r].rung
            path = root / "compile" / mode / rung / "libtinyaf.so"
            if not path.is_file():
                ap.error(f"Missing {path}; build it with: make MODE={mode} RUNG={rung}")
            lib = Predictor(rung, mode).lib
            lib.tinyaf_default_threshold.restype = ct.c_float
            if lib.tinyaf_default_threshold() != float(np.float32(THRESHOLDS[r])):
                ap.error(f"Stale threshold in {path}; rebuild with: make MODE={mode} RUNG={rung}")
            libraries[f"{mode}/{r}"] = hashfile(path)
    code_paths = [Path(__file__), root / "src/datasets/native.py"]
    checkpoints = {}
    if "python" in args.implementations:
        code_paths.extend(sorted((root / "python_model").rglob("*.py")))
        checkpoints = {
            r: hashfile(MODELS[r].checkpoint) for r in args.models
        }
    if c_modes:
        code_paths.extend(
            root / p for p in ["c_model/tinyaf.c", "c_model/tinyaf.h", "src/bindings.py"]
        )
    asset_metadata = {str(MODELS[r].metadata.relative_to(root)): hashfile(MODELS[r].metadata)
                      for r in args.models}
    manifest = {
        "schema_version": 4,
        "model_implementations": {r: [m for m in args.implementations if m in MODELS[r].implementations]
                                  for r in args.models},
        "devices": {r: devices for r in args.models if python_selected},
        "gpu": gpu_info,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") if gpu_info else None,
        "metadata": asset_metadata,
        "config": config,
        "max_per_record": args.max_per_record,
        "cohorts": args.cohorts,
        "models": args.models,
        "implementations": args.implementations,
        "thresholds": {r: evaluation_threshold(r) for r in args.models},
        "reference": (
            {
                "torch": torch.__version__,
                "threads": 1,
                "mkldnn": False,
                "batch_size": args.nn_batch_size,
                "nn_batch_size": args.nn_batch_size,
                "deterministic": True,
                "tf32": False,
                "cuda": torch.version.cuda if gpu_info else None,
                "cudnn": torch.backends.cudnn.version() if gpu_info else None,
            }
            if "python" in args.implementations else None
        ),
        "inputs": metadata_provenance(config, args.cohorts),
        "code": {str(p.relative_to(root)): hashfile(p) for p in code_paths},
        "libraries": libraries,
        "checkpoints": checkpoints,
    }
    mpth = output / "manifest.json"
    if mpth.is_file() and json.loads(mpth.read_text()) != manifest:
        raise RuntimeError(
            "output belongs to different inputs/code; choose a fresh directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    mpth.write_text(json.dumps(manifest, indent=2) + "\n")
    tasks = []
    for cohort in args.cohorts:
        d = metadata(cohort, config)
        (output / cohort).mkdir(exist_ok=True)
        for i, (key, rows) in enumerate(d.groupby("record_key", sort=True)):
            if args.max_per_record and len(rows) > args.max_per_record:
                rows = rows.iloc[
                    np.linspace(0, len(rows) - 1, args.max_per_record, dtype=int)
                ]
            tasks.append((cohort, key, rows, str(output / cohort / f"{i:05d}.parquet")))
    print("tasks", len(tasks), "windows", sum(len(t[2]) for t in tasks), flush=True)
    start = time.monotonic()
    done = 0
    with ProcessPoolExecutor(
        args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
        initargs=(config, args.models, args.implementations, devices, args.nn_batch_size),
    ) as pool:
        futures = [pool.submit(job, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            n, status, path = f.result()
            done += n
            print(
                f"{i}/{len(tasks)} records; {done} windows; {time.monotonic()-start:.1f}s; {status} {path}",
                flush=True,
            )
    print("COMPLETE", done, flush=True)


if __name__ == "__main__":
    main()
