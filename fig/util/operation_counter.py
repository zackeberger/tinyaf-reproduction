#!/usr/bin/env python3
"""Operation estimates for TinyAF-S, HR+HRV, SmallNN, and LargeNN figures.

Run from the repository root (CPU only; no checkpoints or ECG data needed)::

    python fig/util/operation_counter.py
    python fig/util/operation_counter.py --details
    python fig/util/operation_counter.py --json fig/outputs/operation_counts.json

Compare one 10-second, 256 Hz Quanta ADC input, conditioned to 100 Hz, at
4, 9, and 20 RR intervals.

Shared and specific prep follow the Python reference algorithms. Each scalar
MAC, other arithmetic operation, signal comparison or math primitive counts
as one operation.

Inference uses PyTorch FlopCounterMode's built-in formulas, without custom
handlers. Its convolution and matrix-product FLOPs are divided by two to
retain MAC=1. Unregistered operator calls are reported separately; they are
not scalar operation counts. Bias arithmetic is omitted by the built-in
formulas even when part of a registered operator. Normalization, activations,
pooling and elementwise arithmetic are not counted by PyTorch 2.7.1. Models
return logits; sigmoid/threshold postprocessing is outside the reported scope.

The actual local Python models run in eval mode on float32 CPU inputs with
batch size one. TinyAF-S alone uses FlexibleTinyAF: L real beat encodings plus
one reusable blank embedding when L<20, followed by the unchanged 20-slot
tensor stages. This wrapper implements the accounting optimization without
changing the production model. SmallNN, LargeNN and HR+HRV run their original
forward methods. No weights or ECG downloads are needed for shape-based costs.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
import operator
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode


ROOT = Path(__file__).resolve().parents[2]
FS_IN, FS, SECONDS = 256, 100, 10
N = FS * SECONDS
BEAT_CASES = (4, 9, 20)
METHODS = ("HR+HRV", "TinyAF-S", "SmallNN", "LargeNN")
KINDS = {"mac", "add", "sub", "mul", "div", "sqrt", "exp", "abs", "neg",
         "compare", "finite"}


@dataclass
class Row:
    stage: str
    note: str
    counts: Counter = field(default_factory=Counter)

    def add(self, **counts):
        if counts.keys() - KINDS or any(v < 0 for v in counts.values()):
            raise ValueError(f"Invalid operation counts: {counts}")
        self.counts.update(counts)

    @property
    def total(self):
        return sum(self.counts.values())

    def scaled(self, factor):
        return Row(self.stage, self.note, Counter({k: v * factor for k, v in self.counts.items()}))

    def as_dict(self):
        return dict(stage=self.stage, ops=self.total,
                    by_operation=dict(self.counts), note=self.note)


class Scalar:
    """Evaluate small data-dependent numeric paths while counting their work.

    Values use Python arithmetic to define the fixture, not to emulate FP32
    rounding. Numeric predicates are counted; Python loop/index work is not.
    """

    def __init__(self, value, row):
        self.value = float(value)
        self.row = row

    @staticmethod
    def value_of(value):
        return value.value if isinstance(value, Scalar) else value

    def binary(self, other, kind, function, reverse=False):
        self.row.add(**{kind: 1})
        a, b = self.value, self.value_of(other)
        if reverse:
            a, b = b, a
        return Scalar(function(a, b), self.row)

    def compare(self, other, function):
        self.row.add(compare=1)
        return function(self.value, self.value_of(other))

    def __add__(self, other): return self.binary(other, "add", operator.add)
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self.binary(other, "sub", operator.sub)
    def __rsub__(self, other): return self.binary(other, "sub", operator.sub, True)
    def __mul__(self, other): return self.binary(other, "mul", operator.mul)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self.binary(other, "div", operator.truediv)
    def __lt__(self, other): return self.compare(other, operator.lt)
    def __le__(self, other): return self.compare(other, operator.le)
    def __gt__(self, other): return self.compare(other, operator.gt)
    def __ge__(self, other): return self.compare(other, operator.ge)
    def __eq__(self, other): return self.compare(other, operator.eq)
    def __ne__(self, other): return self.compare(other, operator.ne)

    def __abs__(self):
        self.row.add(abs=1)
        return Scalar(abs(self.value), self.row)

    def finite(self):
        self.row.add(finite=1)
        return math.isfinite(self.value)

    def isnan(self):
        self.row.add(finite=1)
        return math.isnan(self.value)


def mac(a, b, c):
    """Count a multiply-accumulate once, independently of compiler contraction."""
    row = next(v.row for v in (a, b, c) if isinstance(v, Scalar))
    row.add(mac=1)
    return Scalar(Scalar.value_of(a) * Scalar.value_of(b) + Scalar.value_of(c), row)


def numeric_mean(values):
    """NumPy-style mean budget: n-1 additions and one division."""
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total / len(values)


def interpolation(row, n, m, span):
    """np.interp arithmetic on the Python pipeline's np.linspace grids.

    NumPy precomputes all n-1 slopes when n <= m, even when some outputs
    coincide with input samples. Otherwise slopes are computed on demand.
    Grid construction, interval searches and exact-hit tests are excluded.
    Each slope costs two subtractions and one division; each interpolated
    value costs a time-offset subtraction and one MAC. Exact hits are copies.
    """
    if n < 2 or m < 2:
        raise ValueError("Interpolation requires at least two input/output samples")
    source = np.linspace(0.0, span, n)
    target = np.linspace(0.0, span, m)
    indices = np.searchsorted(source, target)
    interpolated = int(np.count_nonzero(source[indices] != target))
    slopes = n - 1 if n <= m else interpolated
    row.add(sub=2 * slopes + interpolated, div=slopes, mac=interpolated)


def shared_rows():
    n = SECONDS * FS_IN
    adc = Row("Conversion to millivolts", "runtime.predict_quanta: raw.astype(float64) * 0.0030303; input validation and casts excluded.")
    adc.add(mul=n)
    filt = Row("Zero-phase low-pass", "SciPy sosfiltfilt: one second-order section, 40 Hz, odd padding 9 per side, two passes. Each sample: 3 MACs + 2 multiplies + 1 add. Padding: 2 multiplies + 18 subtractions; state scaling: 4 multiplies.")
    filt.add(mac=2 * (n + 18) * 3, mul=6 + 2 * (n + 18) * 2,
             sub=18, add=2 * (n + 18))
    resample = Row("Resample 256 to 100 Hz", "NumPy np.interp value/slope arithmetic; time-grid setup and index searches excluded.")
    interpolation(resample, n, N, SECONDS)
    dc = Row("DC blocker", "One subtraction and one MAC for each sample after y[0]=0.")
    dc.add(sub=N - 1, mac=N - 1)
    norm = Row("Z-score", "(x - x.mean()) / (x.std() + 1e-8): std recomputes the mean; three reductions, two centering passes, explicit squaring, square root and epsilon included.")
    norm.add(add=3 * (N - 1) + 1, div=N + 3, sub=2 * N, mul=N, sqrt=1)
    return [adc, filt, resample, dc, norm]


def clean_peaks(beats):
    if not isinstance(beats, int) or not 4 <= beats <= 20:
        raise ValueError("Integer fixtures require 4 through 20 RR intervals")
    return [round(50 + i * 900 / beats) for i in range(beats + 1)]


def peak_search(row, values=(0, 1, 2, 3, 4, 5, 4, 3, 2, 1)):
    """SciPy find_peaks local maxima/prominence on a prescribed QRS.

    Signal comparisons and prominence arithmetic are counted. Index bounds,
    plateau-index arithmetic and warning/input-validation checks are omitted.
    """
    x = [Scalar(v, row) for v in values]
    maxima = []
    i = 1
    while i < len(x) - 1:
        if x[i - 1] < x[i]:
            j = i + 1
            while j < len(x) - 1 and x[j] == x[i]:
                j += 1
            if x[j] < x[i]:
                maxima.append((i + j - 1) // 2)
                i = j
        i += 1
    prominences = []
    for p in maxima:
        v = left = right = x[p]
        k = p
        while k >= 0 and x[k] <= v:
            if x[k] < left:
                left = x[k]
            k -= 1
        k = p
        while k < len(x) and x[k] <= v:
            if x[k] < right:
                right = x[k]
            k += 1
        prominences.append((v - max(left, right)).value)
    # np.argmax on the candidate prominences; one comparison per extra peak.
    row.add(compare=max(0, len(maxima) - 1))
    return maxima, prominences


def detector_rows(peaks):
    p = len(peaks)
    gradient = Row("Gradient magnitude", "np.gradient with unit sample spacing: central differences/division, one-sided endpoint differences, then np.abs.")
    gradient.add(sub=N, div=N - 2, abs=N)
    smooth = Row("Running boxcars", "NeuroKit signal_smooth uses SciPy uniform_filter1d, nearest boundary mode, widths 10 and 75; initial sum from zero, then one subtraction/addition and division per update.")
    smooth.add(add=10 + 75 + 2 * (N - 1), sub=2 * (N - 1), div=2 * N)
    threshold = Row("QRS threshold", "NeuroKit: 1.5 * average gradient and smoothgrad > threshold; Boolean edge bookkeeping excluded.")
    threshold.add(mul=N, compare=N)
    segments = Row("QRS selection", "NeuroKit: QRS-end filtering, mean-length threshold, one ten-sample QRS per peak, SciPy find_peaks/prominence, argmax and minimum-delay test; no rejected candidates.")
    segments.add(sub=3 * p, add=2 * p - 1, div=1, mul=1, compare=3 * p)
    for _ in peaks:
        peak_search(segments)
    return [gradient, smooth, threshold, segments]


def quantile(values, q):
    """Pandas linear quantile arithmetic on an already ordered window."""
    row = values[0].row
    pos = Scalar(len(values) - 1, row) * q
    lo = int(pos.value)
    if pos.value == lo:  # Rank/index selection is outside the arithmetic metric.
        return values[lo]
    hi = min(lo + 1, len(values) - 1)
    return mac(values[hi] - values[lo], pos - lo, values[lo])


def rolling_stat(values, width, q=None):
    """Centered, min_periods=1 rolling quantile/median with ordered updates.

    Pandas uses skip lists. Binary searches here give a deterministic estimate
    of signal-value comparisons for the same insert/remove operations; list
    movement and index/container bookkeeping are excluded. Each quantile call
    maintains its own ordered set, as in NeuroKit's separate q1/q3 calls.
    """
    result = []
    window = []
    previous_start = previous_end = 0
    for i in range(len(values)):
        start, end = max(0, i - width // 2), min(len(values), i + width // 2 + 1)
        for value in values[previous_end:end]:
            if value.finite():
                window.insert(bisect_left(window, value), value)
        for value in values[previous_start:start]:
            if value.finite():
                window.pop(bisect_left(window, value))
        previous_start, previous_end = start, end
        if not window:
            result.append(Scalar(float("nan"), values[0].row))
        elif q is not None:
            result.append(quantile(window, q))
        else:
            middle = len(window) // 2
            result.append(window[middle] if len(window) % 2 else
                          (window[middle - 1] + window[middle]) / 2)
    return result


def rolling(values, width, threshold):
    """NeuroKit _compute_threshold or the separate rolling median."""
    if not threshold:
        return rolling_stat(values, width)
    absolute = [abs(value) for value in values]
    q1 = rolling_stat(absolute, width, 0.25)
    q3 = rolling_stat(absolute, width, 0.75)
    return [5.2 * ((high - low) / 2) for low, high in zip(q1, q3)]


def artifact_pass(peaks):
    """Mirror NeuroKit2 _find_artifacts numerical expressions for one pass.

    Return both a ledger and the computed subspaces for comparison with the
    actual library in tests. The fixture must require no corrections. NumPy
    logical_and evaluates both arguments; its arithmetic is not short-circuited.
    """
    row = Row("RR artifact checking", "NeuroKit Kubios/Lipponen checks RR timing for extra/missed peaks, ectopic-like patterns and long/short intervals; rolling quartiles (91), median (11), and numerical rules. Ordered-set comparisons are a binary-search estimate of pandas skip-list work.")
    p = len(peaks)
    x = [Scalar(v, row) for v in peaks]
    rr = [v / FS for v in [Scalar(0, row)] + [x[i] - x[i - 1] for i in range(1, p)]]
    rr[0] = numeric_mean(rr[1:])
    d = [Scalar(0, row)] + [rr[i] - rr[i - 1] for i in range(1, p)]
    d[0] = numeric_mean(d[1:])

    def normalize(values, thresholds):
        normalized = [a / b if b != 0 else a for a, b in zip(values, thresholds)]
        for i, threshold in enumerate(thresholds):
            if threshold == 0:  # Separate NumPy mask after np.divide(where=...).
                normalized[i] = Scalar(float("nan"), row)
        return normalized

    th1 = rolling(d, 91, True)
    d = normalize(d, th1)

    def reflect(i):
        while i < 0 or i >= p:
            i = -i if i < 0 else 2 * p - 2 - i
        return i

    def extrema(a, b, function):
        row.add(compare=1)  # Two-element np.min/np.max; NaNs propagate.
        if math.isnan(a.value) or math.isnan(b.value):
            return Scalar(float("nan"), row)
        return Scalar(function(a.value, b.value), row)

    s12, s22 = [], []
    for i in range(p):
        a = b = Scalar(0, row)
        if d[i] > 0:
            a = extrema(d[reflect(i - 1)], d[reflect(i + 1)], max)
        elif d[i] < 0:
            a = extrema(d[reflect(i - 1)], d[reflect(i + 1)], min)
        if d[i] >= 0:
            b = extrema(d[reflect(i + 1)], d[reflect(i + 2)], min)
        elif d[i] < 0:
            b = extrema(d[reflect(i + 1)], d[reflect(i + 2)], max)
        s12.append(a)
        s22.append(b)
    med = rolling(rr, 11, False)
    m = [a - b for a, b in zip(rr, med)]
    negative = [a < 0 for a in m]
    row.add(compare=p)  # The Python indexing expression evaluates this mask twice.
    m = [a * 2 if is_negative else a for a, is_negative in zip(m, negative)]
    th2 = rolling(m, 91, True)
    m = normalize(m, th2)
    i = 0
    while i < p - 2:
        if abs(d[i]) <= 1:
            i += 1
            continue
        eq1a, eq1b = d[i] > 1, s12[i] < mac(-0.13, d[i], -0.17)
        eq2a, eq2b = d[i] < -1, s12[i] > mac(-0.13, d[i], 0.17)
        if (eq1a and eq1b) or (eq2a and eq2b):
            raise ValueError("Clean fixture unexpectedly requires ectopic correction")
        large_d, large_m = abs(d[i]) > 1, abs(m[i]) > 3
        if not (large_d or large_m):
            i += 1
            continue
        candidates = (i, i + 1) if abs(d[i + 1]) < abs(d[i + 2]) else (i,)
        for j in candidates:
            eq3a, eq3b = d[j] > 1, s22[j] < -1
            eq4 = abs(m[j]) > 3
            eq5a, eq5b = d[j] < -1, s22[j] > 1
            if (eq3a and eq3b) or eq4 or (eq5a and eq5b):
                raise ValueError("Clean fixture unexpectedly requires artifact correction")
            i += 1
    return row, {key: np.array([v.value for v in values]) for key, values in
                 (('rr', rr), ('drrs', d), ('mrrs', m), ('s12', s12), ('s22', s22))}


def kubios_row(peaks):
    """Two passes: initial check, then the iterative stopping check; no edits."""
    row, _ = artifact_pass(peaks)
    row.note += " Two identical passes, zero detected artifacts and zero corrections."
    return row.scaled(2)


def beat_rows(peaks):
    l = len(peaks) - 1
    select = Row("Beat intervals and RR", "pipeline.stack_beats: in-window peak mask, interval validity and length; predict_one divides all 20 RR slots by sampling frequency.")
    select.add(compare=2 * len(peaks) + l, sub=2 * l, div=20)
    resample = Row("Beat resampling", "pipeline.stack_beats: np.interp to 100 samples for each real interval; NumPy slope precomputation included, grids and searches excluded.")
    for a, b in zip(peaks, peaks[1:]):
        interpolation(resample, b - a, 100, 1)
    return [select, resample]


def hrv_rows(peaks):
    l = len(peaks) - 1
    valid = Row("In-window peaks", "Use the full peak list for MeanNN and sample SDNN.")
    valid.add(compare=2 * len(peaks))
    features = Row("MeanNN and SDNN", "Required two-feature NumPy calculation: diff(peaks) * (1000/fs), mean and std(ddof=1). std recomputes the mean; finite-input arithmetic only, unrelated HRV features excluded.")
    features.add(sub=2 * l, mul=2 * l, add=3 * (l - 1), div=3, sqrt=1)
    return [valid, features]


class FlexibleTinyAF(torch.nn.Module):
    """Profile reusable blank embeddings while preserving Python model outputs."""

    def __init__(self, model, rr_intervals):
        super().__init__()
        self.model = model
        self.encoded = min(rr_intervals + 1, model.n_beats)

    def forward(self, beats, rr_seconds):
        net, batch = self.model, beats.shape[0]
        h = beats[:, :self.encoded].reshape(batch * self.encoded, 1, net.beat_w)
        h = F.relu(net.conv1(h))
        h = F.relu(net.conv2(h))
        h = F.relu(net.beat_enc(h.reshape(batch * self.encoded, net.per_beat)))
        h = h.reshape(batch, self.encoded, net.embed)
        if self.encoded < net.n_beats:
            blank = h[:, -1:].expand(batch, net.n_beats - self.encoded, net.embed)
            h = torch.cat((h, blank), dim=1)
        rr = (rr_seconds - net.rr_mean) / (net.rr_std + 1e-8)
        h = torch.cat((h, rr.unsqueeze(-1)), dim=-1)
        h = F.relu(net.beat_conv(h.transpose(1, 2)))
        live = (rr_seconds != 0).to(h.dtype).unsqueeze(1)
        bins = net.fc1.in_features // net.beat_ch
        num = F.adaptive_avg_pool1d(h * live, bins)
        den = F.adaptive_avg_pool1d(live, bins)
        h = (num / den.clamp_min(1e-6)).reshape(batch, net.fc1.in_features)
        return net.fc(F.relu(net.fc1(h)))


class OperatorCalls(TorchDispatchMode):
    """Observe executed operators; all numerical cost formulas remain in PyTorch."""

    def __init__(self):
        super().__init__()
        self.calls = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[func._overloadpacket] += 1
        return func(*args, **(kwargs or {}))


@lru_cache(maxsize=4)
def inference_model(method):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from python_model.models.tinyaf import TinyAF
    from python_model.models.hrhrv import HRHRV
    from python_model.models.smallnn import SmallNN
    from python_model.models.largenn import LargeNN

    factories = {"TinyAF-S": lambda: TinyAF.rung("S"), "HR+HRV": HRHRV,
                 "SmallNN": SmallNN, "LargeNN": LargeNN}
    if method not in factories:
        raise ValueError(f"Unknown method: {method}")
    # Shapes determine costs; initialization is outside the counting context.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return factories[method]().cpu().eval()


def inference_inputs(method, beats):
    # Build the same input tensors as the Python pipeline, outside inference.
    from python_model.pipeline import stack_beats, hr_hrv_ms

    peaks = np.array(clean_peaks(beats))
    trace = np.linspace(-1, 1, N, dtype=np.float32)
    if method == "TinyAF-S":
        waveforms, rr, _ = stack_beats(trace, FS, peaks=peaks)
        return (torch.from_numpy(waveforms).unsqueeze(0),
                torch.from_numpy(rr / FS).unsqueeze(0))
    if method == "HR+HRV":
        return (torch.from_numpy(hr_hrv_ms(peaks, FS)).unsqueeze(0),)
    return (torch.from_numpy(trace).unsqueeze(0),)


@lru_cache(maxsize=None)
def inference_profile(method, beats):
    net = inference_model(method)
    inputs = inference_inputs(method, beats)
    if method == "TinyAF-S":
        net = FlexibleTinyAF(net, beats).eval()
    observed = OperatorCalls()
    # Place the observer below FlopCounterMode so it sees decomposed operators.
    with torch.no_grad(), observed, FlopCounterMode(display=False) as counter:
        net(*inputs)
    flops = {str(op): n for op, n in counter.get_flop_counts()["Global"].items()}
    # Only these formulas represent 2 FLOPs per MAC. Fail rather than silently
    # halving a different kind of work if PyTorch's built-in coverage changes.
    mac_ops = {"aten.convolution", "aten._convolution", "aten.addmm", "aten.mm",
               "aten.bmm", "aten.baddbmm"}
    if flops.keys() - mac_ops or any(n % 2 for n in flops.values()):
        raise RuntimeError(f"Reaudit FLOP-to-MAC conversion for PyTorch {torch.__version__}: {flops}")
    return dict(
        method=method, rr_intervals=beats,
        model_class=f"{net.__class__.__module__}.{net.__class__.__name__}",
        input_shapes=[list(x.shape) for x in inputs],
        encoded_beats=min(beats + 1, 20) if method == "TinyAF-S" else None,
        flops=counter.get_total_flops(), operations=counter.get_total_flops() // 2,
        flops_by_operator=flops,
        unregistered_operator_calls={str(op): n for op, n in
                                     sorted(observed.calls.items(), key=lambda item: str(item[0]))
                                     if op not in counter.flop_registry},
    )


def inference_rows(method, beats):
    profile = inference_profile(method, beats)
    return [Row(op, "FlopCounterMode built-in formula; 2 FLOPs converted to 1 MAC; bias excluded.",
                Counter(mac=n // 2))
            for op, n in sorted(profile["flops_by_operator"].items())]


def integer_budget(method, beats):
    peaks = clean_peaks(beats)
    shared = shared_rows()
    specific = []
    if method in ("TinyAF-S", "HR+HRV"):
        specific = detector_rows(peaks) + [kubios_row(peaks)]
        specific += beat_rows(peaks) if method == "TinyAF-S" else hrv_rows(peaks)
    inference = inference_rows(method, beats)
    return dict(shared=shared, specific=specific, inference=inference)


def count_method(method, beats):
    """Return the full ledger, interpolating complete integer cases if needed."""
    if not math.isfinite(beats) or not 4 <= beats <= 20:
        raise ValueError("RR intervals must be finite and between 4 and 20")
    lo, hi = math.floor(beats), math.ceil(beats)
    groups = integer_budget(method, lo)
    if lo != hi:
        upper = integer_budget(method, hi)
        weight = beats - lo
        for key, rows in groups.items():
            for a, b in zip(rows, upper[key], strict=True):
                if a.stage != b.stage:
                    raise ValueError("Cannot interpolate different accounting stages")
                a.counts = Counter({k: a.counts[k] * (1 - weight) + b.counts[k] * weight
                                    for k in a.counts.keys() | b.counts.keys()})
    totals = {key: sum(row.total for row in rows) for key, rows in groups.items()}
    return dict(method=method, beats=beats, rr_intervals=beats, peaks=beats + 1,
                **totals, total=sum(totals.values()),
                stages={key: [row.as_dict() for row in rows] for key, rows in groups.items()})


def report():
    results = [count_method(method, beats) for method in METHODS for beats in BEAT_CASES]
    ranges = []
    for method in METHODS:
        cases = [result for result in results if result["method"] == method]
        minimum, maximum = min(r["total"] for r in cases), max(r["total"] for r in cases)
        ranges.append(dict(method=method, minimum=minimum, maximum=maximum,
                           minimum_rr_intervals=[r["rr_intervals"] for r in cases if r["total"] == minimum],
                           maximum_rr_intervals=[r["rr_intervals"] for r in cases if r["total"] == maximum]))
    return dict(
        schema_version=3, implementation="Python prep and FlopCounterMode inference with flexible TinyAF encoding",
        fs_in=FS_IN, fs_model=FS, seconds=SECONDS,
        beat_cases=list(BEAT_CASES), rr_interval_cases=list(BEAT_CASES),
        convention="One scalar MAC, other arithmetic operation, signal-value comparison or math primitive = one operation; algorithmic estimate, not CPU cycles.",
        inference_counter=dict(
            library="torch.utils.flop_counter.FlopCounterMode",
            torch_version=torch.__version__, device="cpu", dtype="float32",
            batch_size=1, mode="eval", custom_handlers=False,
            convention="Convolution/matrix-product FLOPs divided by 2; MAC=1.",
            exclusions=[
                "Bias arithmetic inside convolution/linear operators.",
                "Unregistered normalization, activation, pooling, elementwise, comparison and validity operations.",
                "Tensor movement, shape/index operations, allocation and copies.",
                "Model setup and post-logit sigmoid/threshold conversion.",
            ],
            profiles=[inference_profile(method, beats) for method in METHODS for beats in BEAT_CASES],
        ),
        references=[
            "python_model/runtime.py: predict_quanta (mV conversion).",
            "python_model/pipeline.py: condition, detect_r_peaks, stack_beats, hr_hrv_ms; runtime._window_arrays.",
            "python_model/models/tinyaf.py: TinyAF.rung('S'), executed through FlexibleTinyAF for padding reuse.",
            "NeuroKit2 0.2.12: _ecg_findpeaks_neurokit, _find_artifacts, _compute_threshold, _signal_fixpeaks_kubios.",
            "NumPy np.interp/mean/std; SciPy sosfiltfilt/uniform_filter1d/find_peaks; pandas centered rolling quantiles/median.",
            "python_model/models/{smallnn,largenn,hrhrv}.py: actual forward methods.",
            "torch.utils.flop_counter: built-in convolution and matrix-product formulas.",
        ],
        assumptions=[
            "Quanta ADC-to-mV scaling and shared Python conditioning for every method; no device-specific centering.",
            "P=L+1 clean peaks, evenly spaced over 0.5..9.5 s and rounded at 100 Hz; one 10-sample QRS each.",
            "L=9 is a 60-bpm reference fixture (10 peaks, nine one-second RR intervals), not a measured average workload.",
            "RR artifact checking is NeuroKit's Kubios/Lipponen peak-timing analysis, not general ECG quality scoring: two passes, no corrections in these fixtures.",
            "Rolling-statistic numerical arithmetic is counted; binary-search ordered-set comparisons estimate pandas skip-list searches, not exact backend instructions.",
            "TinyAF-S assumes L real beat encodings plus one reusable padding embedding for L<20. This is an explicit optimization estimate; the current Python forward encodes all 20 slots.",
            "TinyAF retains Python zero-padding convolution MACs; the full 20-slot stages execute, with unregistered operations excluded from inference counts.",
            "All models run in eval mode, batch size one, float32 CPU; no batch-norm folding, no active dropout. Unregistered operations receive no inference cost.",
            "HR+HRV counts the required mean RR and sample SD of RR features, not unrelated features of a full HRV report.",
            "Inference ends at the logit. Bias, unregistered inference operations and post-logit sigmoid/threshold are excluded; totals are partial numerical-work estimates, not complete FLOP counts.",
            "Prep excludes input validation, Boolean control flow, address/index bookkeeping, interpolation searches, casts/copies, memory traffic and coefficient/time-grid/fixed-scale setup.",
            "Min/max summarize only the evaluated 4, 9 and 20 RR-interval fixtures; they are not universal bounds or measured cohort averages.",
        ],
        results=results, ranges=ranges,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--details", action="store_true", help="Print each counted stage and operation breakdown")
    parser.add_argument("--json", type=Path, help="Also write the full ledger and assumptions as JSON")
    args = parser.parse_args()
    data = report()
    print("Python-reference operations per 10-second ECG, 256 Hz input; MAC=1.")
    print(f"Inference: FlopCounterMode (PyTorch {torch.__version__}); convolution/matrix FLOPs / 2.")
    print("TinyAF-S assumes flexible beat encoding with reuse of one padding embedding.")
    print("Clean-peak estimates at 4, 9, and 20 RR intervals. Not CPU cycles.\n")
    print(f"{'Method':<10}{'RR':>8}{'Peaks':>8}{'Shared prep':>14}{'Specific prep':>15}{'Inference':>14}{'Total':>15}")
    for result in data["results"]:
        print(f"{result['method']:<10}{result['beats']:>8g}{result['peaks']:>8g}"
              + "".join(f"{result[key]:>{width},.1f}" for key, width in
                        (("shared", 14), ("specific", 15), ("inference", 14), ("total", 15))))
        if args.details:
            for group, rows in result["stages"].items():
                for row in rows:
                    counts = ", ".join(f"{k}={v:g}" for k, v in sorted(row["by_operation"].items()) if v)
                    print(f"  {group}: {row['stage']}: {row['ops']:,.3f} [{counts}]")
                    print(f"    {row['note']}")
    print("\nTotal-operation ranges over the three evaluated RR-interval cases:")
    for result in data["ranges"]:
        print(f"{result['method']:<10} {result['minimum']:>15,.1f} -- {result['maximum']:>15,.1f}")
    print("\nInference exclusions:")
    for exclusion in data["inference_counter"]["exclusions"]:
        print(f"  - {exclusion}")
    if args.details:
        print("\nUnregistered inference operators (call counts, not scalar operation counts):")
        for profile in data["inference_counter"]["profiles"]:
            calls = ", ".join(f"{op}={n}" for op, n in profile["unregistered_operator_calls"].items())
            print(f"  {profile['method']}, RR={profile['rr_intervals']}: {calls}")
    print("\nAssumptions:")
    for assumption in data["assumptions"]:
        print(f"  - {assumption}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
