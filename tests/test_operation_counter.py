"""Validate accounting against the Python/SciPy/NeuroKit/PyTorch references."""

import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import signal

from fig.util import operation_counter as counter


def test_shared_budget_matches_documented_python_arithmetic():
    rows = {row.stage: row for row in counter.shared_rows()}
    assert {name: row.total for name, row in rows.items()} == {
        "Conversion to millivolts": 2560,
        "Zero-phase low-pass": 30960,
        "Resample 256 to 100 Hz": 4990,
        "DC blocker": 1998,
        "Z-score": 7002,
    }
    assert sum(row.total for row in rows.values()) == 47510
    assert rows["Conversion to millivolts"].counts == {"mul": 2560}
    # x.mean(), std's internal mean, and sum of squared deviations are separate.
    assert rows["Z-score"].counts["add"] == 3 * 999 + 1
    assert rows["Z-score"].counts["mac"] == 0


def test_filter_budget_matches_actual_scipy_pass_lengths(monkeypatch):
    module = importlib.import_module("scipy.signal._signaltools")
    original = module.sosfilt
    lengths = []

    def record(sos, data, *args, **kwargs):
        lengths.append(len(data))
        assert sos.shape == (1, 6)
        return original(sos, data, *args, **kwargs)

    monkeypatch.setattr(module, "sosfilt", record)
    sos = signal.butter(2, 40, fs=256, output="sos")
    signal.sosfiltfilt(sos, np.zeros(2560))
    assert lengths == [2578, 2578]
    row = next(r for r in counter.shared_rows() if r.stage == "Zero-phase low-pass")
    # Six units per biquad update; two endpoint multiplications, eighteen
    # padding subtractions, and four initial-state scaling multiplications.
    assert row.total == 6 * sum(lengths) + 2 + 18 + 4


@pytest.mark.parametrize("n,m,expected", [(2560, 1000, 4990), (100, 100, 297), (2, 100, 199)])
def test_numpy_interpolation_precomputes_slopes_when_upsampling(n, m, expected):
    row = counter.Row("interpolation", "")
    counter.interpolation(row, n, m, 10.0 if n == 2560 else 1.0)
    assert row.total == expected
    if n == m:
        # np.interp still builds all slopes although every output is a copy.
        assert row.counts["mac"] == 0
        assert row.counts["div"] == n - 1


@pytest.mark.parametrize("values", [
    [0, 1, 2, 3, 4, 5, 4, 3, 2, 1],
    [0, 3, 3, 3, 0],
    [0, 3, 1, 2, 0, 4, 0],
    [1, 1, 1, 1],
])
def test_peak_search_matches_scipy(values):
    row = counter.Row("peak search", "")
    indices, prominences = counter.peak_search(row, values)
    actual, properties = signal.find_peaks(values, prominence=(None, None))
    np.testing.assert_array_equal(indices, actual)
    np.testing.assert_allclose(prominences, properties["prominences"])


@pytest.mark.parametrize("width", [3, 11, 91])
def test_rolling_statistics_match_pandas_at_edges_and_missing_values(width):
    values = [1, float("nan"), 3, 3, -2, 5, 8, 0, 4, -1, 6, 6]
    row = counter.Row("rolling", "")
    wrapped = [counter.Scalar(value, row) for value in values]
    window = pd.Series(values).rolling(width, center=True, min_periods=1)
    actual_median = window.median().to_numpy()
    median = counter.rolling(wrapped, width, False)
    np.testing.assert_allclose([v.value for v in median], actual_median, equal_nan=True)
    absolute = pd.Series(values).abs().rolling(width, center=True, min_periods=1)
    actual_threshold = 5.2 * (absolute.quantile(0.75) - absolute.quantile(0.25)) / 2
    threshold = counter.rolling(wrapped, width, True)
    np.testing.assert_allclose([v.value for v in threshold], actual_threshold, equal_nan=True)
    assert row.counts["compare"] > 0


@pytest.mark.parametrize("method,convolution,linear", [
    ("SmallNN", 1_200_460, 24),
    ("LargeNN", 436_346_112, 5_120),
    ("HR+HRV", 0, 2),
])
def test_library_counts_match_independent_baseline_mac_references(method, convolution, linear):
    # Saved fvcore convolution/linear counts from research c0738ad; BN is now
    # outside the built-in inference counter's coverage.
    profile = counter.inference_profile(method, 4)
    by_op = profile["flops_by_operator"]
    assert by_op.get("aten.convolution", 0) == 2 * convolution
    assert by_op["aten.addmm"] == 2 * linear
    assert profile["operations"] == convolution + linear
    assert sum(row.total for row in counter.inference_rows(method, 4)) == convolution + linear
    omitted = profile["unregistered_operator_calls"]
    assert ("aten.div" if method == "HR+HRV"
            else "aten._native_batch_norm_legit_no_training") in omitted


@pytest.mark.parametrize("beats", [4, 9, 19, 20])
def test_flexible_tinyaf_preserves_output_and_library_counts_actual_encodings(beats):
    net = counter.inference_model("TinyAF-S")
    inputs = counter.inference_inputs("TinyAF-S", beats)
    with torch.no_grad():
        reference = net(*inputs)
    encoded = min(beats + 1, net.n_beats)
    executed = {}
    handles = []

    def capture(name):
        def hook(layer, args, output):
            executed[name] = output.numel() * layer.weight[0].numel()
            if name in ("conv1", "conv2", "beat_enc"):
                assert args[0].shape[0] == encoded
        return hook

    for name, layer in net.named_modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(capture(name)))
    try:
        with torch.no_grad():
            actual = counter.FlexibleTinyAF(net, beats).eval()(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    profile = counter.inference_profile("TinyAF-S", beats)
    assert profile["encoded_beats"] == encoded
    assert profile["operations"] == sum(executed.values())
    assert profile["flops"] == 2 * sum(executed.values())
    assert profile["input_shapes"] == [[1, 20, 100], [1, 20]]
    assert {"aten.relu", "aten.div", "aten._adaptive_avg_pool2d"} <= (
        profile["unregistered_operator_calls"].keys())


def test_new_library_cost_conventions_require_explicit_review(monkeypatch):
    class DifferentCounter(counter.FlopCounterMode):
        def get_flop_counts(self):
            return {"Global": {torch.ops.aten.add: 7}}

    monkeypatch.setattr(counter, "FlopCounterMode", DifferentCounter)
    # Call the uncached function so it executes the altered library contract.
    with pytest.raises(RuntimeError, match="Reaudit FLOP-to-MAC"):
        counter.inference_profile.__wrapped__("HR+HRV", 4)


@pytest.mark.parametrize("method", counter.METHODS)
def test_fractional_case_interpolates_complete_ledgers(method):
    low = counter.count_method(method, 11)
    high = counter.count_method(method, 12)
    mean = counter.count_method(method, 11.861)
    for key in ("shared", "specific", "inference", "total"):
        assert mean[key] == pytest.approx(0.139 * low[key] + 0.861 * high[key])
    assert mean["peaks"] == 12.861
    assert mean["total"] == pytest.approx(mean["shared"] + mean["specific"] + mean["inference"])


@pytest.mark.parametrize("beats", [4, 9, 11, 12, 19, 20])
def test_artifact_accounting_matches_neurokit_and_two_pass_stop(monkeypatch, beats):
    module = importlib.import_module("neurokit2.signal.signal_fixpeaks")
    original = module._find_artifacts
    peaks = np.array(counter.clean_peaks(beats))
    artifacts, subspaces = original(peaks, sampling_rate=100)
    assert not any(artifacts.values())
    one_pass, estimated = counter.artifact_pass(peaks)
    for name, values in estimated.items():
        np.testing.assert_allclose(values, subspaces[name], equal_nan=True, rtol=1e-10, atol=1e-12)
    calls = []

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_find_artifacts", record)
    _, corrected = module._signal_fixpeaks_kubios(peaks, sampling_rate=100)
    np.testing.assert_array_equal(corrected, peaks)
    assert len(calls) == 2
    row = counter.kubios_row(peaks)
    assert row.total == 2 * one_pass.total
    assert row.counts["compare"] > 0
    assert row.counts["finite"] > 0


def test_cli_exports_balanced_json_from_outside_repository(tmp_path):
    script = Path(counter.__file__).resolve()
    output = tmp_path / "nested" / "counts.json"
    result = subprocess.run([sys.executable, str(script), "--json", str(output)],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    assert "4, 9, and 20 RR intervals" in result.stdout
    data = json.loads(output.read_text())
    assert data["fs_in"] == 256
    assert data["beat_cases"] == [4, 9, 20]
    assert data["rr_interval_cases"] == [4, 9, 20]
    assert data["schema_version"] == 3
    assert data["implementation"] == "Python prep and FlopCounterMode inference with flexible TinyAF encoding"
    library = data["inference_counter"]
    assert library["torch_version"] == torch.__version__
    assert library["custom_handlers"] is False
    assert library["batch_size"] == 1
    assert len(library["profiles"]) == 12
    assert library["exclusions"]
    for profile in library["profiles"]:
        assert profile["flops"] == sum(profile["flops_by_operator"].values())
        assert profile["operations"] * 2 == profile["flops"]
        assert profile["unregistered_operator_calls"]
        budget = next(r for r in data["results"] if
                      (r["method"], r["rr_intervals"]) ==
                      (profile["method"], profile["rr_intervals"]))
        assert budget["inference"] == profile["operations"]
    assert "Inference exclusions:" in result.stdout
    assert len(data["results"]) == 12
    for result in data["results"]:
        for group, rows in result["stages"].items():
            assert result[group] == pytest.approx(sum(row["ops"] for row in rows))
            for row in rows:
                assert row["ops"] == pytest.approx(sum(row["by_operation"].values()))
                assert all(v >= 0 for v in row["by_operation"].values())
        assert result["total"] == pytest.approx(sum(result[k] for k in ("shared", "specific", "inference")))
        assert result["rr_intervals"] == result["beats"]
    for method in ("SmallNN", "LargeNN"):
        totals = {r["total"] for r in data["results"] if r["method"] == method}
        assert len(totals) == 1
    for method in counter.METHODS:
        unique_costs = {r["inference"] for r in data["results"] if r["method"] == method}
        assert len(unique_costs) == (3 if method == "TinyAF-S" else 1)
    for item in data["ranges"]:
        totals = [r["total"] for r in data["results"] if r["method"] == item["method"]]
        assert (item["minimum"], item["maximum"]) == (min(totals), max(totals))


def test_manuscript_table_matches_report():
    manuscript = Path(counter.__file__).with_suffix(".tex").read_text()
    lines = [line.split("&") for line in manuscript.splitlines() if "&" in line]
    data = counter.report()
    for method in counter.METHODS:
        row = next(line for line in lines if line[0].strip() == method)
        values = [cell.strip().removesuffix(r"\\").strip() for cell in row[1:]]
        cases = {r["rr_intervals"]: r["total"] for r in data["results"]
                 if r["method"] == method}
        assert values == [f"{cases[beats]:,}" for beats in counter.BEAT_CASES]
    assert f"{data['results'][0]['shared']:,}" in manuscript
    assert "optimized C implementation" not in manuscript
    assert "FlopCounterMode" in manuscript
    assert "2.7.1" in manuscript
    for row in counter.shared_rows():
        assert f"{row.total:,}" in manuscript
    for profile in data["inference_counter"]["profiles"]:
        assert f"{profile['operations']:,}" in manuscript


@pytest.mark.parametrize("beats", [0, 3, 21, float("nan"), float("inf")])
def test_reject_invalid_beat_counts(beats):
    with pytest.raises(ValueError, match="between 4 and 20"):
        counter.count_method("TinyAF-S", beats)
