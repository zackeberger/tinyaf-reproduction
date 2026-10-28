"""Run model families on example ECG arrays in Python. C is enabled for TinyAF.
"""

import subprocess

import numpy as np
from scipy.special import expit

from python_model.runtime import (
    MODELS, TINYAF_MODELS, model_name, configure_reference, load_model, predict_quanta,
    predict_one,
)

from .bindings import Predictor, ROOT


def predict_python(ecg_array, model="TINYAF-S", sample_rate=256, quanta_adc_to_physical=False, device="cpu"):
    """Predict AF/AFL probabilities with reproducible Python inference.

    Parameters
    ----------
    ecg_array : array_like, shape (samples,) or (N, samples)
        Ten-second ECG windows in physical units, such as millivolts. When
        ``quanta_adc_to_physical=True``, each window must contain 2,560 integer
        ADC counts in 0..4095, sampled at 256 Hz.
    model : str, optional
        TINYAF-S/M/L, LARGENN, SMALLNN, or HRHRV (case-insensitive).
        Default is TINYAF-S.
    sample_rate : float, optional
        Sampling rate in Hz for physical inputs. Default is 256. Ignored for
        Quanta ADC inputs, which use 256 Hz and 0.0030303 mV per count.
    quanta_adc_to_physical : bool, optional
        Convert Quanta ADC counts to millivolts before preprocessing.
        Default is False.
    device : str, optional
        PyTorch device for Python inference (default cpu).

    Returns
    -------
    numpy.ndarray, shape (N, 1), dtype float32
        Probabilities in input order. NaN indicates an inference failure.

    Notes
    -----
    Loads the selected checkpoint once per call and reuses it for all N windows.
    Sets one Torch thread, disables MKLDNN, and requires deterministic algorithms
    for the current process.
    """
    configure_reference()
    net = load_model(model, device=device)
    x = np.asarray(ecg_array)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("expected one window or an array of windows")
    result = []
    for raw in x:
        logit = (
            predict_quanta(raw, net)
            if quanta_adc_to_physical
            else predict_one(raw, sample_rate, net)
        )
        result.append(expit(logit))
    return np.asarray(result, dtype=np.float32).reshape(-1, 1)


def compile_c_shared_library(model="TINYAF-S", mode="optimized"):
    """Run make for a TinyAF model and mode reference/optimized; return its .so path.

    Builds the shared library and example executable in ``compile/<mode>/<model>``.
    Existing targets are reused when current. Requires make and a C compiler.
    """
    model = model_name(model)
    if model not in TINYAF_MODELS:
        raise ValueError("C supports TINYAF-S, TINYAF-M, or TINYAF-L")
    if mode not in ["reference", "optimized"]:
        raise ValueError("unknown C mode")
    rung = MODELS[model].rung
    subprocess.run(["make", f"RUNG={rung}", f"MODE={mode}"], cwd=ROOT, check=True)
    return ROOT / "compile" / mode / rung / "libtinyaf.so"


def predict_c(ecg_array, model="TINYAF-S", threshold=-1, mode="optimized"):
    """Build and run TinyAF C on one or more Quanta ADC windows.

    Parameters
    ----------
    ecg_array : array_like, shape (2560,) or (N, 2560)
        Ten-second, 256 Hz ECG windows containing integer ADC counts in 0..4095.
    model : str, optional
        TINYAF-S, TINYAF-M, or TINYAF-L. Default is TINYAF-S.
    threshold : float, optional
        Classification threshold in [0, 1], or -1 for the model default.
        Passed to C and validated there; it does not change the probability.
    mode : str, optional
        C implementation: reference or optimized. Default is optimized.

    Returns
    -------
    numpy.ndarray, shape (N, 1), dtype float32
        Probabilities in input order. NaN indicates no usable beat interval.

    Raises
    ------
    ValueError
        If the model, mode, input shape, dtype, or ADC range is invalid.
    RuntimeError
        If C returns a failure other than no usable beats.
    """
    model = model_name(model)
    compile_c_shared_library(model, mode)
    c = Predictor(MODELS[model].rung, mode)
    x = np.asarray(ecg_array)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 2560:
        raise ValueError("expected 2560 ADC counts per window")
    if not np.issubdtype(x.dtype, np.integer) or np.any(x < 0) or np.any(x > 4095):
        raise ValueError("expected integer ADC counts in 0..4095")
    out = []
    for raw in x:
        status, pred = c.adc(raw, threshold)
        if status not in [0, -4]:
            raise RuntimeError(f"TinyAF C failed with status {status}")
        out.append(pred.probability)
    return np.asarray(out, np.float32).reshape(-1, 1)
