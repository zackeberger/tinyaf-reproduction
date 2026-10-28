"""Load TinyAF and baseline models and score raw or prepared ECG windows.

Preprocessing runs on CPU. Model tensors use each model's device, with support
for individual windows and batches of prepared windows.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np
import torch

from . import pipeline
from .models.tinyaf import TinyAF
from .models.largenn import LargeNN
from .models.smallnn import SmallNN
from .models.hrhrv import HRHRV

ROOT_ASSETS = Path(__file__).resolve().parents[1] / "model_assets"
ASSETS = ROOT_ASSETS / "tinyaf"
TINYAF_MODELS = ("TINYAF-S", "TINYAF-M", "TINYAF-L")
NN_MODELS = ("LARGENN", "SMALLNN")
MODEL_NAMES = (*TINYAF_MODELS, *NN_MODELS, "HRHRV")


@dataclass(frozen=True)
class ModelSpec:
    """Describe a bundled model's assets and supported inputs/implementations.

    Parameters
    ----------
    family : str
        Subdirectory under ``model_assets``.
    stem : str
        Filename stem for the checkpoint and metadata.
    input_kind : str
        ``"beats"``, ``"trace"``, or ``"hrhrv"``.
    implementations : tuple of str
        Supported implementations: Python, reference C, and/or optimized C.
    rung : str or None, optional
        TinyAF size (S, M, or L); None for baseline models.
    """

    family: str
    stem: str
    input_kind: str
    implementations: tuple
    rung: str | None = None

    @property
    def checkpoint(self):
        """Path to the bundled PyTorch state dictionary."""
        return ROOT_ASSETS / self.family / (self.stem + ".pth")

    @property
    def metadata(self):
        """Path to the model's JSON metadata, including its threshold."""
        return ROOT_ASSETS / self.family / (self.stem + "_meta.json")


MODELS = {
    **{r: ModelSpec("tinyaf", r.replace("-", "_"), "beats",
                    ("python", "reference", "optimized"), rung=r[-1]) for r in TINYAF_MODELS},
    **{r: ModelSpec(r.lower(), r, "trace", ("python",)) for r in NN_MODELS},
    "HRHRV": ModelSpec("hrhrv", "HRHRV", "hrhrv", ("python",)),
}


def model_name(name):
    """Validate a full model name, ignoring case.

    Parameters
    ----------
    name : str
        Full model name from ``MODEL_NAMES``.

    Returns
    -------
    str
        Canonical uppercase name.

    Raises
    ------
    ValueError
        If the name is unknown.
    """
    key = str(name).upper()
    if key not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return key


class _Thresholds(Mapping):
    """Read only selected metadata, so unrelated assets need not be installed."""

    def __getitem__(self, key):
        """Read the selected model's threshold from its metadata."""
        return json.loads(MODELS[model_name(key)].metadata.read_text())["threshold"]

    def __iter__(self):
        return iter(MODELS)

    def __len__(self):
        return len(MODELS)


THRESHOLDS = _Thresholds()


def evaluation_threshold(name):
    """Return the saved threshold at the model's evaluation precision.

    Parameters
    ----------
    name : str
        Full model name, ignoring case.

    Returns
    -------
    float
        TinyAF thresholds rounded to float32 to match C deployment;
        baseline thresholds retain their saved precision.
    """
    key = model_name(name)
    threshold = THRESHOLDS[key]
    return float(np.float32(threshold)) if key in TINYAF_MODELS else threshold


def configure_reference():
    """Configure deterministic inference and single-threaded PyTorch CPU work.

    Disable MKLDNN, TF32, and cuDNN benchmarking, and enable deterministic
    algorithms. These settings affect the current process.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_model(name="TINYAF-S", device="cpu"):
    """Strictly load a bundled checkpoint and return an eval-mode model.

    Parameters
    ----------
    name : str, optional
        Full model name, ignoring case. Default is ``"TINYAF-S"``.
    device : str or torch.device, optional
        Destination for model parameters and buffers. Default is ``"cpu"``.

    Returns
    -------
    torch.nn.Module
        Model with trained weights on the requested device, in evaluation mode.

    Raises
    ------
    ValueError
        If the model name is unknown.
    FileNotFoundError
        If the bundled checkpoint is missing.
    RuntimeError
        If checkpoint parameters are incompatible or the device is unavailable.
    """
    key = model_name(name)
    net = (TinyAF.rung(MODELS[key].rung) if key in TINYAF_MODELS else
           {"LARGENN": LargeNN, "SMALLNN": SmallNN, "HRHRV": HRHRV}[key]())
    state = torch.load(MODELS[key].checkpoint, map_location="cpu", weights_only=True)
    if key in TINYAF_MODELS:
        # These released TinyAF buffers are unused by the beat/RR architecture.
        state.pop("hr_mean", None)
        state.pop("hr_std", None)
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


@dataclass
class Prediction:
    """One model's score and preprocessing/inference status.

    Attributes
    ----------
    logit : float
        Score before sigmoid conversion; NaN when no prediction is available.
    status : int
        Zero on success. Failures: -2 invalid window/rate, -4 no usable beats,
        -6 nonfinite input, -9 preprocessing/inference error, or -12 undefined
        HR/HRV features.
    error : str
        Diagnostic text when available; otherwise an empty string.
    """

    logit: float = float("nan")
    status: int = 0
    error: str = ""


def _prediction_for_window(window, kind):
    """Initialize a result and reject unusable features for the input kind."""
    pred = Prediction(status=window.status, error=window.error)
    if pred.status:
        return pred
    if kind == "beats" and (window.peak_error or window.beat_error or not window.n_beats):
        pred.error = window.peak_error or window.beat_error
        pred.status = -9 if pred.error else -4
    if kind == "hrhrv":
        if window.peak_error:
            pred.status, pred.error = -9, window.peak_error
        elif not np.isfinite(window.hrhrv).all():
            missing = [name for name, value in zip(("mean RR", "SDNN"), window.hrhrv)
                       if not np.isfinite(value)]
            pred.status, pred.error = -12, "undefined HRHRV features: " + ", ".join(missing)
    return pred


def _window_arrays(window, kind):
    """Select model inputs, converting TinyAF RR lengths from samples to seconds."""
    return ({
        "trace": lambda: (window.trace,),
        "hrhrv": lambda: (window.hrhrv,),
        "beats": lambda: (window.beats, window.rr / pipeline.TARGET_FS),
    }[kind])()


def predict_prepared(window, models):
    """Score transient features independently for each selected model.

    Parameters
    ----------
    window : pipeline.PreparedWindow
        Features prepared for every selected model's ``input_kind``.
    models : mapping of str to torch.nn.Module
        Output keys and loaded models, each with an ``input_kind`` and parameters
        on its inference device.

    Returns
    -------
    dict of str to Prediction
        One result per model, including failures.

    Notes
    -----
    Inputs have batch size one and are shared by input kind and device. Models
    run in evaluation mode without gradients. Exceptions and nonfinite outputs
    become failures for that model without suppressing other predictions.
    """
    results = {}
    inputs = {}
    with torch.inference_mode():
        for name, net in models.items():
            kind = net.input_kind
            pred = _prediction_for_window(window, kind)
            results[name] = pred
            if pred.status:
                continue
            try:
                device = next(net.parameters()).device
                cache_key = (kind, str(device))
                if cache_key not in inputs:
                    arrays = _window_arrays(window, kind)
                    inputs[cache_key] = tuple(
                        torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                        .unsqueeze(0).to(device) for a in arrays
                    )
                net.eval()
                value = float(net(*inputs[cache_key]).reshape(-1)[0].item())
                if not np.isfinite(value):
                    raise ValueError("nonfinite model output")
                pred.logit = value
            except Exception as exc:
                pred.status, pred.error = -9, type(exc).__name__ + ": " + str(exc)[:160]
    return results


def predict_models(raw, fs, models):
    """Score selected models on one raw ECG window with shared preprocessing.

    Parameters
    ----------
    raw : array_like, shape (10 * fs,)
        Ten seconds of single-lead ECG samples in physical units.
    fs : float
        Native sampling frequency in Hz.
    models : mapping of str to torch.nn.Module
        Output keys and loaded models with an ``input_kind`` attribute.

    Returns
    -------
    dict of str to Prediction
        Per-model logits and statuses. CPU preprocessing runs once for the
        requested input kinds; inference uses each model's device.
    """
    window = pipeline.prepare_window(raw, fs, {m.input_kind for m in models.values()})
    return predict_prepared(window, models)


def predict_batch(windows, models):
    """Score prepared windows for any model family, preserving row order.

    Parameters
    ----------
    windows : sequence of pipeline.PreparedWindow
        Windows prepared for every selected model's ``input_kind``.
    models : mapping of str to torch.nn.Module
        Output keys and loaded models with an ``input_kind`` attribute.

    Returns
    -------
    list of dict of str to Prediction
        Per-window model results in input order, retaining failed rows.

    Notes
    -----
    Valid rows and input tensors are shared by input kind and device. A batch
    exception triggers individual retries for that model. Nonfinite outputs
    mark only the affected rows as failures.
    """
    results = [dict() for _ in windows]
    inputs = {}
    with torch.inference_mode():
        for name, net in models.items():
            kind = net.input_kind
            for i, window in enumerate(windows):
                results[i][name] = _prediction_for_window(window, kind)
            valid = [i for i, result in enumerate(results) if result[name].status == 0]
            if not valid:
                continue
            try:
                device = next(net.parameters()).device
                cache_key = (kind, str(device))
                if cache_key not in inputs:
                    arrays = [_window_arrays(windows[i], kind) for i in valid]
                    inputs[cache_key] = tuple(
                        torch.from_numpy(np.stack(values).astype(np.float32)).to(device)
                        for values in zip(*arrays)
                    )
                net.eval()
                values = net(*inputs[cache_key]).reshape(-1).detach().cpu().numpy()
                if len(values) != len(valid):
                    raise ValueError("unexpected model output shape")
                for i, value in zip(valid, values):
                    if np.isfinite(value):
                        results[i][name].logit = float(value)
                    else:
                        pred = results[i][name]
                        pred.status, pred.error = -9, "ValueError: nonfinite model output"
            except Exception:
                for i in valid:
                    results[i][name] = predict_prepared(windows[i], {name: net})[name]
    return results


def predict_window(raw, fs, net):
    """Score one raw ECG window with a loaded model.

    Parameters
    ----------
    raw : array_like, shape (10 * fs,)
        Ten seconds of single-lead ECG samples in physical units.
    fs : float
        Native sampling frequency in Hz.
    net : torch.nn.Module
        Loaded model with an ``input_kind`` attribute, on its inference device.

    Returns
    -------
    Prediction
        Logit, status, and diagnostic text for this window.
    """
    return predict_models(raw, fs, {"model": net})["model"]


def predict_one(raw, fs, net):
    """Return one model logit from a raw physical ECG; NaN on inference failure.

    Parameters
    ----------
    raw : array_like, shape (10 * fs,)
        Ten seconds of single-lead ECG samples in physical units.
    fs : float
        Native sampling frequency in Hz.
    net : torch.nn.Module
        Loaded model with an ``input_kind`` attribute, on its inference device.

    Returns
    -------
    float
        Score before sigmoid conversion, or NaN on failure. Use
        ``predict_window`` for the accompanying status and error.
    """
    return predict_window(raw, fs, net).logit


def predict_quanta(raw, net):
    """Validate Quanta ADC counts, convert to mV, and return one logit.

    Parameters
    ----------
    raw : array_like, shape (2560,), integer dtype
        Ten seconds at 256 Hz, with ADC counts in 0..4095.
    net : torch.nn.Module
        Loaded model with an ``input_kind`` attribute, on its inference device.

    Returns
    -------
    float
        Score before sigmoid conversion, or NaN on preprocessing/inference
        failure. ADC counts are converted at 0.0030303 mV per count.

    Raises
    ------
    ValueError
        If the input shape, dtype, or count range is invalid.
    """
    raw = np.asarray(raw)
    if raw.shape != (2560,) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("expected 2560 integer ADC counts")
    if np.any(raw < 0) or np.any(raw > 4095):
        raise ValueError("ADC counts must be in 0..4095")
    return predict_window(raw.astype(np.float64) * 0.0030303, 256, net).logit
