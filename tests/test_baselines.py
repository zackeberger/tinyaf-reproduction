"""Baseline input contracts, independent failure handling, and asset provenance."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from python_model import pipeline
from python_model.runtime import (
    MODELS, NN_MODELS, THRESHOLDS, configure_reference, load_model, predict_models,
    predict_batch,
)
from scripts import evaluate
from scripts.summarize import summarize

configure_reference()


@pytest.mark.parametrize("name,parameters", [("LARGENN", 6385041), ("SMALLNN", 8669), ("HRHRV", 3)])
def test_checkpoint_contract(name, parameters):
    spec = MODELS[name]
    provenance = json.loads((spec.checkpoint.parent / "provenance.json").read_text())
    meta = json.loads(spec.metadata.read_text())
    record = provenance["models"][name]
    assert hashlib.sha256(spec.checkpoint.read_bytes()).hexdigest() == record["sha256"]
    assert THRESHOLDS[name] == meta["threshold"] == record["threshold"]
    assert "original_training_threshold" in meta
    net = load_model(name)
    assert not net.training
    assert sum(p.numel() for p in net.parameters()) == parameters
    width = 2 if name == "HRHRV" else 1000
    with torch.inference_mode():
        assert net(torch.zeros(1, width)).shape == (1, 1)
        with pytest.raises(ValueError, match="expected"):
            net(torch.zeros(1, width + 2))


@pytest.mark.parametrize("features", [[float("nan"), 20.0], [900.0, float("inf")],
                                      [float("nan"), float("nan")]])
def test_hrhrv_rejects_undefined_features(features):
    net = load_model("HRHRV")
    with torch.inference_mode(), pytest.raises(ValueError, match="finite mean RR and SDNN"):
        net(torch.tensor([features]))


def test_hrhrv_uses_full_peak_list_and_sample_std():
    peaks = np.cumsum([0] + [25] * 22 + [100, 125])
    rri = np.diff(peaks).astype(np.float64) / 100 * 1000
    np.testing.assert_array_equal(pipeline.hr_hrv_ms(peaks),
                                  np.float32([rri.mean(), rri.std(ddof=1)]))
    assert pipeline.hr_hrv_ms(peaks)[0] != pipeline.hr_hrv_ms(peaks[:21])[0]
    assert np.isnan(pipeline.hr_hrv_ms([])).all()
    assert np.isnan(pipeline.hr_hrv_ms([20, 95])[1])
    assert pipeline.hr_hrv_ms([20, 95])[0] == 750


@pytest.mark.parametrize("detector_raises", [False, True])
def test_detector_failure_does_not_suppress_other_models(monkeypatch, detector_raises):
    calls = []

    def detector(*args):
        calls.append(1)
        if detector_raises:
            raise RuntimeError("detector failed")
        return np.empty(0, dtype=np.int64)

    monkeypatch.setattr(pipeline, "detect_r_peaks", detector)
    models = {name: load_model(name) for name in ["TINYAF-S", *NN_MODELS, "HRHRV"]}
    predictions = predict_models(np.zeros(2000), 200, models)
    assert len(calls) == 1
    assert predictions["TINYAF-S"].status == (-9 if detector_raises else -4)
    assert np.isnan(predictions["TINYAF-S"].logit)
    for name in NN_MODELS:
        assert predictions[name].status == 0
        assert np.isfinite(predictions[name].logit)
    hr = predictions["HRHRV"]
    assert hr.status == (-9 if detector_raises else -12)
    assert np.isnan(hr.logit)
    assert hr.error == ("RuntimeError: detector failed" if detector_raises
                        else "undefined HRHRV features: mean RR, SDNN")


@pytest.mark.parametrize("peaks,hr_status", [([20, 95], -12), ([20, 95, 180], 0)])
def test_hrhrv_requires_two_intervals(monkeypatch, peaks, hr_status):
    monkeypatch.setattr(pipeline, "detect_r_peaks", lambda *args: np.array(peaks))
    models = {name: load_model(name) for name in ["TINYAF-S", "HRHRV"]}
    predictions = predict_models(np.zeros(2000), 200, models)
    assert predictions["TINYAF-S"].status == 0
    hr = predictions["HRHRV"]
    assert hr.status == hr_status
    assert np.isfinite(hr.logit) == (hr_status == 0)
    if hr_status:
        assert hr.error == "undefined HRHRV features: SDNN"


def test_flat_models_do_not_run_detector(monkeypatch):
    def forbidden(*args):
        raise AssertionError("flat networks must not depend on peak detection")

    monkeypatch.setattr(pipeline, "detect_r_peaks", forbidden)
    result = predict_models(np.zeros(2000), 200, {"SMALLNN": load_model("SMALLNN")})
    assert result["SMALLNN"].status == 0


def test_bad_raw_input_fails_all_models():
    models = {name: load_model(name) for name in ["TINYAF-S", "SMALLNN", "HRHRV"]}
    raw = np.zeros(2000)
    raw[5] = np.nan
    result = predict_models(raw, 200, models)
    assert all(p.status == -6 and np.isnan(p.logit) for p in result.values())


def test_model_exception_is_independent():
    class Broken(torch.nn.Module):
        input_kind = "trace"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            raise RuntimeError("broken model")

    result = predict_models(np.zeros(2000), 200,
                            {"broken": Broken(), "SMALLNN": load_model("SMALLNN")})
    assert result["broken"].status == -9
    assert result["SMALLNN"].status == 0


def test_beat_extraction_exception_is_independent(monkeypatch):
    monkeypatch.setattr(pipeline, "detect_r_peaks", lambda *args: np.array([20, 95, 180]))

    def broken(*args, **kwargs):
        raise RuntimeError("beat extraction failed")

    monkeypatch.setattr(pipeline, "stack_beats", broken)
    nets = {name: load_model(name) for name in ["TINYAF-S", "SMALLNN", "HRHRV"]}
    result = predict_models(np.zeros(2000), 200, nets)
    assert result["TINYAF-S"].status == -9
    assert result["SMALLNN"].status == result["HRHRV"].status == 0


def test_summary_keeps_independent_denominators(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "models": ["TINYAF-S", "HRHRV"], "thresholds": {"TINYAF-S": .4, "HRHRV": .1},
    }))
    (tmp_path / "SHDB").mkdir()
    pd.DataFrame({
        "global_index": [0, 1, 2], "eligible": [True] * 3, "label": [0, 1, 1],
        "python_TINYAF-S": np.array([np.nan, 2, 2], dtype=np.float32), "python_status_TINYAF-S": [-4, 0, 0],
        "python_HRHRV": np.array([np.nan, np.nan, 2], dtype=np.float32), "python_status_HRHRV": [-12, -12, 0],
    }).to_parquet(tmp_path / "SHDB/00000.parquet", index=False)
    models = summarize(tmp_path)["SHDB"]["models"]
    tiny = models["TINYAF-S"]["implementations"]["python"]
    hr = models["HRHRV"]["implementations"]["python"]
    assert tiny["valid"] == 2 and tiny["failed"] == 1
    assert tiny["coverage"] == 2 / 3
    assert hr["valid"] == 1 and hr["failed"] == 2 and hr["coverage"] == 1 / 3
    assert hr["status_counts"] == {"-12": 2, "0": 1}
    assert hr["operating_point"]["tp"] == 1
    assert hr["operating_point"]["tn"] == 0


def test_trace_batch_preserves_failures_and_order():
    class Limited(torch.nn.Module):
        input_kind = "trace"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            if x.shape[0] > 1 or x[0, 0] == 2:
                raise RuntimeError("unsupported batch or example")
            return x[:, :1] * self.weight

    windows = [pipeline.PreparedWindow(trace=np.full(1000, i, np.float32)) for i in range(4)]
    windows[1].status, windows[1].error = -6, "bad raw input"
    scores = predict_batch(windows, {"limited": Limited(), "SMALLNN": load_model("SMALLNN")})
    assert [s["limited"].status for s in scores] == [0, -6, -9, 0]
    assert [scores[i]["limited"].logit for i in (0, 3)] == [0, 3]
    assert [s["SMALLNN"].status for s in scores] == [0, -6, 0, 0]


@pytest.mark.parametrize("names", [["SMALLNN"], ["TINYAF-S", "TINYAF-M", "TINYAF-L", "HRHRV", "SMALLNN"]])
def test_evaluator_flushes_partial_batch(tmp_path, monkeypatch, names):
    evaluate.initialize({}, names, ["python"], nn_batch_size=2)
    monkeypatch.setattr(pipeline, "detect_r_peaks", lambda *args: np.arange(50, 950, 75))
    signal = np.sin(np.arange(10000) * .1)
    monkeypatch.setattr(evaluate, "read_record", lambda *args: (signal, 200))
    rows = pd.DataFrame({"global_index": [8, 4, 9, 2, 6, 1], "label": 0, "eligible": True,
                         "patient_id": "subject_1",
                         "start_sample": [0, 2000, 12000, 4000, 6000, 8000],
                         "end_sample": [2000, 4000, 14000, 6000, 8000, 10000], "orig_fs": 200})
    sizes = []

    def observed(windows, models):
        sizes.append(len(windows))
        assert set(models) == set(names)
        return predict_batch(windows, models)

    monkeypatch.setattr(evaluate, "predict_batch", observed)
    output = tmp_path / "predictions.parquet"
    evaluate.job(("SHDB", "record", rows, str(output)))
    result = pd.read_parquet(output)
    assert sizes == [2, 2, 1]
    assert result.global_index.tolist() == rows.global_index.tolist()
    for name in names:
        assert result[f"python_status_{name}"].tolist() == [0, 0, -2, 0, 0, 0]
    for row in result.to_dict("records"):
        if row["python_status_SMALLNN"] == 0:
            raw = signal[row["start_sample"]:row["start_sample"] + 2000]
            expected = predict_models(raw, 200, evaluate.NETS)
            for name in names:
                assert row[f"python_{name}"] == pytest.approx(expected[name].logit, abs=1e-5)
