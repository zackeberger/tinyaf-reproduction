"""Exercise evaluation subsets through the CLI with only their required assets."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import wfdb

from python_model.runtime import MODELS, MODEL_NAMES
from scripts.evaluate import main
from scripts.summarize import summarize

ROOT = Path(__file__).resolve().parents[1]


def run(project, module, *args, success=True):
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=project,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
             "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
    return result


def make_project(project, models, implementations, cohorts):
    for directory in ["scripts", "src", "python_model"]:
        shutil.copytree(ROOT / directory, project / directory,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for model in models:
        spec = MODELS[model]
        directory = project / "model_assets" / spec.family
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec.metadata, directory / spec.metadata.name)
        if "python" in implementations:
            shutil.copy2(spec.checkpoint, directory / spec.checkpoint.name)
    c_modes = [m for m in implementations if m != "python"]
    if c_modes:
        (project / "c_model").mkdir()
        for name in ["tinyaf.c", "tinyaf.h"]:
            shutil.copy2(ROOT / "c_model" / name, project / "c_model" / name)
    for mode in c_modes:
        for model in models:
            if mode not in MODELS[model].implementations:
                continue
            relative = Path("compile") / mode / MODELS[model].rung / "libtinyaf.so"
            (project / relative).parent.mkdir(parents=True)
            shutil.copy2(ROOT / relative, project / relative)

    source = project / "raw"
    source.mkdir()
    if "SHDB" in cohorts:
        pd.DataFrame({"Data_ID": ["001"], "Subject_ID": ["subject_shdb"]}).to_csv(
            source / "AdditionalData.csv", index=False)
    if "IRIDIA" in cohorts:
        pd.DataFrame({"record_id": ["record_000"], "patient_id": ["subject_iridia"]}).to_csv(
            source / "iridia-af-metadata-v1.0.1.csv", index=False)
    fs = 200
    t = np.arange(20 * fs) / fs
    signal = 0.05 * np.sin(2 * np.pi * t * 0.2)
    for peak in np.arange(0.5, 20, 0.61):
        signal += np.exp(-(((t - peak) / 0.015) ** 2))
    wfdb.wrsamp("001", fs, ["mV"], ["ECG"], p_signal=signal[:, None],
                fmt=["16"], write_dir=str(source))
    if "IRIDIA" in cohorts:
        (source / "record_000").mkdir()
        with h5py.File(source / "record_000/record_000_ecg_00.h5", "w") as f:
            f["ecg"] = signal[:, None]
    config = {"datasets": {}}
    for cohort in cohorts:
        destination = project / "metadata" / cohort
        destination.mkdir(parents=True)
        # Two valid windows plus an out-of-bounds window that must stay visible.
        rows = pd.DataFrame({
            "global_index": [0, 1, 2], "record_id": ["001"] * 3,
            "start_sample": [0, 2000, 6000], "end_sample": [2000, 4000, 8000],
            "orig_fs": [fs] * 3, "AFIB": [0, 1, 1], "AFL": [0, 0, 0],
        })
        if cohort == "IRIDIA":
            rows["record_id"] = "record_000"
            rows["file_index"] = 0
        if cohort == "CPSC":
            rows["record_key"] = "001"
            rows["window_label"] = ["NSR", "AFIB", "AFL"]
            rows["valid"] = 1
            ineligible = rows.iloc[[0]].copy()
            ineligible["global_index"] = 3
            ineligible["valid"] = 0
            rows = pd.concat([rows, ineligible], ignore_index=True)
        rows.to_parquet(destination / "windows_0000.parquet", index=False)
        config["datasets"][cohort] = {
            "source": str(source), "metadata": str(destination), "native_fs": fs,
        }
    (project / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("models,implementations,cohorts,flags", [
    (["TINYAF-M"], ["python"], ["SHDB"],
     ["--models", "tinyaf-m", "--implementation", "python", "--cohorts", "shdb"]),
    (["TINYAF-S"], ["optimized"], ["SHDB"],
     ["--models", "tinyaf-s", "--implementation", "optimized", "--cohorts", "shdb"]),
    (["TINYAF-L"], ["reference"], ["CPSC"],
     ["--models", "TINYAF-L", "--implementation", "reference", "--cohorts", "CPSC"]),
    (["TINYAF-S", "TINYAF-M"], ["python", "optimized"], ["SHDB", "IRIDIA"],
     ["--models", "TINYAF-M", "TINYAF-S", "TINYAF-S", "--implementation", "optimized", "python",
      "--cohorts", "IRIDIA", "SHDB"]),
    (["TINYAF-S", "TINYAF-M", "TINYAF-L"], ["python", "reference", "optimized"],
     ["SHDB", "IRIDIA", "CPSC"], []),
    (["LARGENN", "SMALLNN", "HRHRV"], ["python"], ["SHDB", "IRIDIA", "CPSC"],
     ["--models", "largenn", "smallnn", "hrhrv", "--implementation", "python"]),
    (["TINYAF-S", "TINYAF-M", "TINYAF-L", "LARGENN", "SMALLNN", "HRHRV"], ["python", "optimized"], ["CPSC"],
     ["--models", "all", "--implementation", "python", "optimized", "--cohorts", "CPSC"]),
], ids=["python-only", "optimized-only", "reference-only", "paired-subset", "defaults",
        "baselines-only", "all-models-mixed-implementations"])
def test_evaluation_selection(tmp_path, models, implementations, cohorts, flags):
    make_project(tmp_path, models, implementations, cohorts)
    arguments = ["--config", "config.json", "--output", "results", "--workers", "1"]
    if "python" in implementations:
        arguments.extend(["--nn-batch-size", "2"])
    run(tmp_path, "scripts.evaluate", *arguments, *flags)
    output = tmp_path / "results"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 4
    assert manifest["models"] == models
    assert manifest["implementations"] == implementations
    assert manifest["cohorts"] == cohorts
    assert set(manifest["thresholds"]) == set(models)
    assert set(manifest["config"]["datasets"]) == set(cohorts)
    assert set(manifest["inputs"]) == set(cohorts)
    assert set(manifest["libraries"]) == {
        f"{mode}/{r}" for mode in implementations if mode != "python" for r in models
        if mode in MODELS[r].implementations
    }
    assert set(manifest["checkpoints"]) == (set(models) if "python" in implementations else set())
    if "python" in implementations:
        assert manifest["reference"]["nn_batch_size"] == 2
        assert manifest["reference"]["batch_size"] == 2
        assert manifest["devices"] == {model: ["cpu"] for model in models}
    for cohort in cohorts:
        predictions = pd.read_parquet(output / cohort / "00000.parquet")
        assert str(predictions.patient_id.dtype) == "string"
        if cohort == "CPSC":
            assert predictions.patient_id.isna().all()
        else:
            assert predictions.patient_id.eq("subject_" + cohort.lower()).all()
            mapping = "AdditionalData.csv" if cohort == "SHDB" else "iridia-af-metadata-v1.0.1.csv"
            assert str(tmp_path / "raw" / mapping) in manifest["inputs"][cohort]
        for model in MODEL_NAMES:
            for mode in ["python", "reference", "optimized"]:
                selected = model in models and mode in implementations and mode in MODELS[model].implementations
                assert (f"{mode}_{model}" in predictions) == selected
                if not selected:
                    continue
                status = f"{mode}_status_{model}"
                assert predictions[status].tolist() == ([0, 0, -2, -10] if cohort == "CPSC" else [0, 0, -2])
                assert predictions[f"{mode}_{model}"].iloc[:2].notna().all()
                assert predictions[f"{mode}_{model}"].iloc[2:].isna().all()
    run(tmp_path, "scripts.summarize", "results")
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary) == set(cohorts)
    for cohort in cohorts:
        assert set(summary[cohort]["models"]) == set(models)
        for model in models:
            entry = summary[cohort]["models"][model]
            assert entry["windows"] == 3
            supported = set(implementations) & set(MODELS[model].implementations)
            assert set(entry["implementations"]) == supported
            assert set(entry["comparisons"]) == (
                supported - {"python"} if "python" in supported else set()
            )
            for stats in entry["implementations"].values():
                assert stats["valid"] == 2
                assert stats["status_counts"] == {"0": 2, "-2": 1}
                assert stats["metrics"]["auroc"] is not None
                assert stats["metrics"]["auprc"] is not None

    if implementations == ["optimized"]:
        # Unselected dataset entries need not exist and do not invalidate a resume.
        config_path = tmp_path / "config.json"
        config = json.loads(config_path.read_text())
        config["datasets"]["CPSC"] = {"metadata": "/missing", "source": "/missing"}
        config_path.write_text(json.dumps(config))
        prediction_path = output / "SHDB/00000.parquet"
        before = prediction_path.stat().st_mtime_ns
        resumed = run(tmp_path, "scripts.evaluate", *arguments, *flags)
        assert "cached" in resumed.stdout
        assert prediction_path.stat().st_mtime_ns == before
        mapping = tmp_path / "raw/AdditionalData.csv"
        mapping.write_text(mapping.read_text().replace("subject_shdb", "changed_subject"))
        rejected = run(tmp_path, "scripts.evaluate", *arguments, *flags, success=False)
        assert "choose a fresh directory" in rejected.stderr
    if not flags:
        rejected = run(tmp_path, "scripts.evaluate", *arguments, "--models", "TINYAF-S", success=False)
        assert "choose a fresh directory" in rejected.stderr


def test_summary_uses_saved_thresholds(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"thresholds": {"TINYAF-S": 0.7}}))
    (tmp_path / "SHDB").mkdir()
    pd.DataFrame({
        "global_index": [0, 1], "eligible": [True, True],
        "label": [0, 1], "python_status": [0, 0], "optimized_status_TINYAF-S": [0, 0],
        "python_TINYAF-S": np.array([0.0, 2.0], dtype=np.float32),
        "optimized_TINYAF-S": np.array([0.0, 2.0], dtype=np.float32),
        "optimized_prob_TINYAF-S": [0.5, 0.880797], "optimized_class_TINYAF-S": [0, 1],
    }).to_parquet(tmp_path / "SHDB/00000.parquet", index=False)
    stats = summarize(tmp_path)["SHDB"]["models"]["TINYAF-S"]
    assert stats["threshold"] == 0.7
    assert stats["comparisons"]["optimized"]["classification_agreement"] == 1.0


@pytest.mark.parametrize("flags,error", [
    (["--models", "S"], "invalid choice: 'S'"),
    (["--models", "m"], "invalid choice: 'M'"),
    (["--models", "L"], "invalid choice: 'L'"),
    (["--model", "S"], "unrecognized arguments: --model S"),
    (["--mode", "python"], "unrecognized arguments: --mode python"),
    (["--datasets", "SHDB"], "unrecognized arguments: --datasets SHDB"),
    (["--implementation", "original"], "invalid choice: 'original'"),
    (["--implementation", "c"], "invalid choice: 'c'"),
])
def test_evaluation_rejects_legacy_aliases(tmp_path, monkeypatch, capsys, flags, error):
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "scripts.evaluate", "--config", str(tmp_path / "missing-config.json"),
        "--output", str(output), *flags,
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert error in capsys.readouterr().err
    assert not output.exists()
