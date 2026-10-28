"""Regression tests for the shipping contract and Python/C preprocessing parity."""

import ctypes as ct
from pathlib import Path
import hashlib, json
import numpy as np
import pytest
from neurokit2 import signal_fixpeaks
from python_model import pipeline
from python_model.runtime import (
    ASSETS,
    configure_reference,
    THRESHOLDS,
)
from src.bindings import Predictor

configure_reference()
ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "data").glob("**/256hz/*.txt"))


@pytest.mark.parametrize("rung", list("SML"))
def test_shipping_threshold_assets(rung):
    meta = json.loads((ASSETS / f"TINYAF_{rung}_meta.json").read_text())
    provenance = json.loads((ASSETS / "provenance.json").read_text())
    selection = json.loads((ASSETS / "threshold_selection.json").read_text())
    assert THRESHOLDS[f"TINYAF-{rung}"] == meta["threshold"] == provenance["models"][f"TINYAF-{rung}"]["threshold"]
    assert THRESHOLDS[f"TINYAF-{rung}"] == selection["models"][f"TINYAF-{rung}"]["threshold"]
    assert selection["target_sensitivity"] == meta["threshold_target_sensitivity"] == 0.85
    assert selection["models"][f"TINYAF-{rung}"]["sensitivity"] >= 0.85
    assert hashlib.sha256((ASSETS / f"TINYAF_{rung}.pth").read_bytes()).hexdigest() == provenance["models"][f"TINYAF-{rung}"]["sha256"]


@pytest.fixture(scope="module")
def reference():
    return Predictor("S", "reference")


@pytest.mark.parametrize("fs", [100, 128, 200, 250, 256])
def test_reference_stages_at_native_rates(reference, fs):
    rng = np.random.default_rng(fs)
    t = np.arange(10 * fs) / fs
    raw = 0.03 * rng.normal(size=len(t)) + 0.3 * np.sin(2 * np.pi * t * 0.2)
    for peak in np.arange(0.5, 10, 0.61):
        raw += np.exp(-(((t - peak) / 0.015) ** 2))
    n, trace, beats, rr, peaks = reference.debug(raw, fs)
    expected = pipeline.condition(raw, fs)
    # Investigate any individual mismatch instead of relaxing the stage check.
    np.testing.assert_array_equal(trace, expected)
    b, r, count = pipeline.stack_beats(expected)
    np.testing.assert_array_equal(beats, b)
    np.testing.assert_array_equal(rr, r)
    assert count > 0


@pytest.mark.parametrize("mode", ["reference", "optimized"])
@pytest.mark.parametrize("rung", list("SML"))
def test_adc_api_errors_defaults_and_workspace(mode, rung):
    c = Predictor(rung, mode)
    entrypoint = c.lib.predict_c
    entrypoint.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags="C_CONTIGUOUS"),
        ct.c_int, ct.c_float, ct.POINTER(ct.c_float),
    ]
    entrypoint.restype = ct.c_int

    def check_adc(raw, threshold=-1):
        status, result = c.adc(raw, threshold)
        probability = ct.c_float(123.0)
        assert entrypoint(raw, len(raw), threshold, ct.byref(probability)) == status
        if status:
            assert np.isnan(probability.value)
        else:
            assert probability.value == result.probability
        return status, result

    raw = np.loadtxt(EXAMPLES[0], dtype=np.uint16)
    status, out = check_adc(raw)
    assert status == 0 and out.ok and out.n_beats > 0
    assert out.threshold == float(np.float32(THRESHOLDS[f"TINYAF-{rung}"]))
    for threshold in [0.0, 1.0, 0.7]:
        status, custom = check_adc(raw, threshold)
        assert status == 0 and custom.logit == out.logit
        assert custom.prediction == (custom.probability >= threshold)
    for threshold in [np.nan, np.inf, -0.5, 1.1]:
        status, invalid = check_adc(raw, threshold)
        assert status == -5 and np.isnan(invalid.probability)
    status, invalid = check_adc(raw[:50])
    assert status == -2 and np.isnan(invalid.logit)
    bad = raw.copy()
    bad[100] = 4096
    status, invalid = check_adc(bad)
    assert status == -3 and np.isnan(invalid.probability)
    status, invalid = check_adc(np.zeros(2560, np.uint16))
    assert status == -4 and not invalid.ok and np.isnan(invalid.logit)
    status, again = check_adc(raw)
    assert status == 0 and again.logit == out.logit
    if mode == "optimized":
        assert c.size + 5120 + 2048 < 40 * 1024


@pytest.mark.parametrize("mode", ["reference", "optimized"])
def test_nonfinite_input_and_bad_rate(mode):
    c = Predictor("S", mode)
    raw = np.zeros(2000)
    raw[22] = np.nan
    status, out = c.native(raw, 200)
    assert status == -6 and np.isnan(out.logit)
    raw[22] = np.inf
    status, out = c.native(raw, 200)
    assert status == -6
    status, out = c.native(np.zeros(2010), 201)
    assert status == -7


def test_kubios_against_neurokit(reference):
    rng = np.random.default_rng(127)
    for _ in range(300):
        n = rng.integers(2, 35)
        peaks = np.cumsum(rng.integers(15, 130, size=n)).astype(np.int32)
        p = np.zeros(128, np.int32)
        p[:n] = peaks
        expected = signal_fixpeaks(peaks, sampling_rate=100, method="Kubios")[1]
        m = reference.lib.tinyaf_debug_correct(p, n, reference.workspace)
        assert m >= 0
        np.testing.assert_array_equal(p[:m], expected)


def test_full_peak_list_before_twenty_beat_limit(reference):
    fs = 256
    t = np.arange(10 * fs) / fs
    raw = np.zeros(len(t))
    for peak in np.arange(0.4, 9.9, 0.32):
        raw += np.exp(-(((t - peak) / 0.014) ** 2))
    n, trace, beats, rr, peaks = reference.debug(raw, fs)
    expected_beats, expected_rr, count = pipeline.stack_beats(
        pipeline.condition(raw, fs)
    )
    assert n > 21 and count == 20
    np.testing.assert_array_equal(beats, expected_beats)
    np.testing.assert_array_equal(rr, expected_rr)
