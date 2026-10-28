"""All-family batching, independent failures, and device selection."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from scipy.special import expit

from python_model import pipeline
from python_model.runtime import (
    MODEL_NAMES, configure_reference, load_model, predict_batch, predict_prepared,
)
from scripts import evaluate, run_quanta_examples as runner


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
)])
def test_all_model_batches_match_single_windows(device):
    configure_reference()
    rng = np.random.default_rng(123)
    windows = []
    for i in range(7):
        trace = rng.normal(size=1000).astype(np.float32)
        peaks = np.arange(50, 950, 60 + i)
        beats, rr, count = pipeline.stack_beats(trace, peaks=peaks)
        windows.append(pipeline.PreparedWindow(
            trace=trace, beats=beats, rr=rr, n_beats=count,
            hrhrv=pipeline.hr_hrv_ms(peaks),
        ))
    windows[1].n_beats = 0
    windows[1].hrhrv[:] = np.nan
    windows[2].peak_error = "detector failed"
    windows[2].hrhrv[1] = np.nan
    windows[3].beat_error = "beat extraction failed"
    windows[4].status, windows[4].error = -6, "nonfinite native input"
    windows[5].status, windows[5].error = -2, "invalid native window or rate"

    models = {name: load_model(name, device) for name in MODEL_NAMES}
    expected = [predict_prepared(window, models) for window in windows]
    calls = {name: [] for name in models}
    handles = []
    for name, net in models.items():
        def observe(module, inputs, name=name):
            assert all(value.device == next(module.parameters()).device for value in inputs)
            calls[name].append(inputs[0].shape[0])
        handles.append(net.register_forward_pre_hook(observe))
    try:
        actual = predict_batch(windows, models)
    finally:
        for handle in handles:
            handle.remove()

    for name, net in models.items():
        assert calls[name] == [{"beats": 2, "hrhrv": 3, "trace": 5}[net.input_kind]]
    for row, baseline in zip(actual, expected):
        for name, pred in row.items():
            reference = baseline[name]
            assert pred.status == reference.status
            assert pred.error == reference.error
            if reference.status:
                assert np.isnan(pred.logit)
            else:
                assert expit(pred.logit) == pytest.approx(expit(reference.logit), abs=1e-4)


@pytest.mark.parametrize("kind", ["beats", "hrhrv"])
def test_batch_failure_retries_windows_and_preserves_status(kind):
    class Limited(torch.nn.Module):
        input_kind = kind

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, x, *args):
            value = x.reshape(len(x), -1)[:, :1]
            if len(x) > 1 or value[0, 0] == 2:
                raise RuntimeError("unsupported batch or example")
            return value * self.weight

    windows = [pipeline.PreparedWindow(
        beats=np.full((20, 100), i, np.float32), rr=np.full(20, 75, np.float32),
        n_beats=20, hrhrv=np.array([i, 20], np.float32),
    ) for i in range(4)]
    windows[1].status, windows[1].error = -6, "bad raw input"
    results = predict_batch(windows, {"limited": Limited(), "HRHRV": load_model("HRHRV")})
    assert [row["limited"].status for row in results] == [0, -6, -9, 0]
    assert [results[i]["limited"].logit for i in (0, 3)] == [0, 3]
    assert [row["HRHRV"].status for row in results] == [0, -6, 0, 0]


def test_nonfinite_batch_output_fails():
    class Nonfinite(torch.nn.Module):
        input_kind = "hrhrv"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((1,), float("nan")))

        def forward(self, x):
            return x[:, :1] * self.weight

    window = pipeline.PreparedWindow(hrhrv=np.array([750, 20], np.float32))
    pred = predict_batch([window], {"bad": Nonfinite()})[0]["bad"]
    assert pred.status == -9 and np.isnan(pred.logit)
    assert pred.error == "ValueError: nonfinite model output"


def test_evaluator_assigns_all_models_to_worker_device(monkeypatch):
    loaded = {}

    def load(name, device):
        loaded[name] = device
        return object()

    monkeypatch.setattr(evaluate, "load_model", load)
    monkeypatch.setattr(evaluate.mp, "current_process", lambda: SimpleNamespace(_identity=(2,)))
    evaluate.initialize({}, MODEL_NAMES, ["python"], ["cuda:0", "cuda:1"], 4)
    assert loaded == {name: "cuda:1" for name in MODEL_NAMES}
    assert set(evaluate.BATCH_MODELS) == set(MODEL_NAMES)


@pytest.mark.parametrize("name", ["TINYAF-S", "HRHRV"])
def test_evaluator_rejects_unavailable_cuda_for_every_family(tmp_path, monkeypatch, capsys, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", [
        "scripts.evaluate", "--config", str(tmp_path / "missing.json"),
        "--output", str(tmp_path / "results"), "--models", name,
        "--implementation", "python", "--nn-device", "cuda:0",
    ])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 2
    assert "CUDA is unavailable" in capsys.readouterr().err


def test_quanta_runner_uses_explicit_names_and_device_for_all_models(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_quanta_examples", "--examples-dir", "unused", "--implementation", "python",
        "--models", "tinyaf-s", "TINYAF-M", "TINYAF-L", "HRHRV", "SMALLNN", "LARGENN",
        "--nn-device", "cuda:1",
    ])
    calls = []

    def predict(name, examples, device):
        calls.append((name, device))
        return []

    monkeypatch.setattr(runner, "load_examples", lambda path: [])
    monkeypatch.setattr(runner, "predict_python_model", predict)
    assert runner.main() == 0
    assert dict(calls) == {name: "cuda:1" for name in MODEL_NAMES}
