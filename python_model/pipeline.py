"""Prepare raw ECG windows for TinyAF and its baseline models.

Condition each window once, then derive only the requested beat/RR or HR/HRV
features. Arrays remain on CPU; model inference is handled by ``runtime``.
"""

import numpy as np
from dataclasses import dataclass
from neurokit2 import ecg_peaks
from scipy.signal import butter, sosfiltfilt

SECONDS = 10
TARGET_FS = 100     # Hz after conditioning
LOWPASS_HZ = 40.0
LOWPASS_ORDER = 2
POLE_A = 0.9686     # DC-blocker pole; corner near 0.5 Hz at 100 Hz
BEAT_W = 100        # Samples per resampled beat
MAX_BEATS = 20


def _dc_block(x, a=POLE_A):
    """Remove DC using ``y[n] = x[n] - x[n-1] + a * y[n-1]``.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_samples,)
        Floating-point signal samples.
    a : float, optional
        Filter pole. Default is 0.9686, giving a corner near 0.5 Hz at 100 Hz.

    Returns
    -------
    y : numpy.ndarray
        Filtered signal with the same shape and dtype as ``x``. The first
        output sample is zero.
    """
    y = np.empty_like(x)
    y[0] = 0.0
    for n in range(1, len(x)):
        y[n] = x[n] - x[n - 1] + a * y[n - 1]
    return y


def condition(raw, fs, seconds=SECONDS, target_fs=TARGET_FS):
    """Filter, resample, and normalize one ECG window.

    Apply a zero-phase, second-order 40 Hz low-pass filter before linear
    resampling, then remove DC and z-score normalize the window.

    Parameters
    ----------
    raw : array_like, shape (n_samples,)
        Single-lead ECG samples in physical units, such as millivolts.
    fs : float
        Input sampling frequency in Hz.
    seconds : float, optional
        Window duration in seconds. Default is 10.
    target_fs : float, optional
        Output sampling frequency in Hz. Default is 100.

    Returns
    -------
    trace : numpy.ndarray, shape (round(seconds * target_fs),), dtype float32
        Conditioned ECG samples.

    Raises
    ------
    ValueError
        If the input length does not match ``round(seconds * fs)``.
    """
    x = np.asarray(raw, dtype=np.float64).reshape(-1)
    n_in = int(round(seconds * fs))
    if len(x) != n_in:
        raise ValueError(f"expected {n_in} samples ({seconds} s at {fs} Hz), got {len(x)}")

    # Filter before downsampling to suppress aliasing.
    sos = butter(LOWPASS_ORDER, LOWPASS_HZ, fs=fs, btype="low", output="sos")
    x = sosfiltfilt(sos, x)

    n_out = int(round(seconds * target_fs))
    if n_out != len(x):
        x = np.interp(np.linspace(0.0, seconds, n_out),
                      np.linspace(0.0, seconds, len(x)), x)

    x = _dc_block(x)
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


def detect_r_peaks(trace, fs, *, method="neurokit", correct_artifacts=True):
    """Detect and correct R peaks, keeping only indices within the ECG window.

    Parameters
    ----------
    trace : array_like, shape (n_samples,)
        Conditioned single-lead ECG samples, converted to float64 for detection.
    fs : float
        Sampling frequency of ``trace`` in Hz.
    method : str, optional
        NeuroKit2 peak-detection method. Default is ``"neurokit"``.
    correct_artifacts : bool, optional
        Apply Kubios artifact correction to the full peak list.
        Default is ``True``.

    Returns
    -------
    peaks : numpy.ndarray, shape (n_peaks,), dtype int64
        Peak indices satisfying ``0 <= peak < len(trace)``, in their original
        order. May be empty.

    Notes
    -----
    When enabled, artifact correction precedes filtering peaks to window bounds.
    This function does not limit the number of peaks.
    """
    x = np.asarray(trace, dtype=np.float64)
    _, info = ecg_peaks(x, sampling_rate=fs, method=method,
                        correct_artifacts=correct_artifacts)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    peaks = peaks[(peaks >= 0) & (peaks < len(x))]
    return peaks


def stack_beats(trace, fs=TARGET_FS, *, width=BEAT_W,
                max_beats=MAX_BEATS, peaks=None):
    """Extract fixed-length beat waveforms and RR intervals from an ECG trace.

    Use supplied peaks or detect and correct them before selecting up to
    ``max_beats`` R-to-R intervals. Skip intervals shorter than two samples.

    Parameters
    ----------
    trace : array_like, shape (n_samples,)
        Conditioned ECG samples.
    fs : float, optional
        Sampling frequency of ``trace`` in Hz, used for peak detection when
        ``peaks`` is None. Default is 100.
    width : int, optional
        Number of samples per resampled beat. Default is 100.
    max_beats : int, optional
        Maximum number of beat slots. Default is 20.
    peaks : array_like, shape (n_peaks,), optional
        Precomputed, corrected peak indices in chronological order and within
        ``trace`` bounds. Used directly, without detection or correction.
        Default is None, which calls ``detect_r_peaks``.

    Returns
    -------
    beats : numpy.ndarray, shape (max_beats, width), dtype float32
        Beat waveforms in chronological order, linearly resampled to
        ``width`` samples. Unused rows are zero-padded.
    rr : numpy.ndarray, shape (max_beats,), dtype float32
        Interval lengths in samples at ``fs`` Hz. Unused entries are zero.
    n_beats : int
        Number of populated slots. Zero if no valid interval is found.
    """
    x = np.asarray(trace)
    r = detect_r_peaks(x, fs) if peaks is None else np.asarray(peaks, dtype=np.int64)

    beats = np.zeros((max_beats, width), dtype=np.float32)
    rr = np.zeros(max_beats, dtype=np.float32)
    grid = np.linspace(0.0, 1.0, width)
    k = 0
    for j in range(len(r) - 1):
        a, b = int(r[j]), int(r[j + 1])
        if b - a < 2 or k >= max_beats:
            continue
        seg = x[a:b]
        beats[k] = np.interp(grid, np.linspace(0.0, 1.0, len(seg)), seg)
        rr[k] = b - a
        k += 1
    return beats, rr, k



def hr_hrv_ms(peaks, fs=TARGET_FS):
    """Mean RR and sample SDNN in ms, from the full corrected peak list.

    Parameters
    ----------
    peaks : array_like, shape (n_peaks,)
        Corrected R-peak sample indices in chronological order.
    fs : float, optional
        Sampling frequency of the peak indices in Hz. Default is 100.

    Returns
    -------
    features : numpy.ndarray, shape (2,), dtype float32
        Mean RR and sample standard deviation of all supplied RR intervals
        (SDNN, with ``ddof=1``), in milliseconds. Mean RR requires two peaks;
        SDNN requires three. Undefined features remain NaN.
    """
    rri = np.diff(np.asarray(peaks, dtype=np.float64)) / fs * 1000.0
    mean = rri.mean() if len(rri) >= 1 else np.nan
    sd = rri.std(ddof=1) if len(rri) >= 2 else np.nan
    return np.asarray([mean, sd], dtype=np.float32)


@dataclass
class PreparedWindow:
    """Transient CPU features and preprocessing status for one ECG window.

    Attributes
    ----------
    trace : numpy.ndarray, shape (1000,), or None
        Conditioned float32 ECG at 100 Hz; None if conditioning did not finish.
    peaks : numpy.ndarray, shape (n_peaks,), or None
        Corrected peak indices at 100 Hz; None when detection was not requested.
    beats : numpy.ndarray, shape (20, 100), or None
        Float32 beat waveforms with unused slots zero-padded.
    rr : numpy.ndarray, shape (20,), or None
        Float32 RR lengths in samples at 100 Hz, with unused entries zero.
    n_beats : int
        Number of populated beat slots.
    hrhrv : numpy.ndarray, shape (2,), or None
        Float32 mean RR and SDNN in ms, with NaNs for undefined features.
        None when HR/HRV features were not requested.
    status, error : int, str
        Shared input/conditioning status and diagnostic text; zero means success.
    peak_error, beat_error : str
        Independent detection/extraction errors; these leave the trace usable.
    """

    trace: object = None
    peaks: object = None
    beats: object = None
    rr: object = None
    n_beats: int = 0
    hrhrv: object = None
    status: int = 0
    error: str = ""
    peak_error: str = ""
    beat_error: str = ""


def prepare_window(raw, fs, input_kinds):
    """Condition once, and detect peaks only when a selected model needs them.

    Parameters
    ----------
    raw : array_like, shape (10 * fs,)
        Ten seconds of single-lead ECG samples in physical units.
    fs : float
        Native sampling frequency in Hz; must be finite and greater than 80.
    input_kinds : collection of str
        Requested inputs: ``"trace"``, ``"beats"``, and/or ``"hrhrv"``.
        Peak detection is shared by the beat and HR/HRV paths.

    Returns
    -------
    PreparedWindow
        Conditioned trace and requested features. Invalid input or conditioning
        failures set ``status`` and ``error``. Detection and beat-extraction
        errors are recorded separately so trace models can still run.

    Notes
    -----
    Missing beats and undefined HR/HRV features are retained here; ``runtime``
    turns them into model-specific failures.
    """
    window = PreparedWindow()
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(fs) or fs <= 80 or len(raw) != round(SECONDS * fs):
        window.status, window.error = -2, "invalid native window or rate"
        return window
    if not np.isfinite(raw).all():
        window.status, window.error = -6, "nonfinite native input"
        return window
    try:
        window.trace = condition(raw, fs)
        if not np.isfinite(window.trace).all():
            raise ValueError("nonfinite conditioned trace")
    except Exception as exc:
        window.status, window.error = -9, type(exc).__name__ + ": " + str(exc)[:160]
        return window
    if not ({"beats", "hrhrv"} & set(input_kinds)):
        return window
    try:
        window.peaks = detect_r_peaks(window.trace, TARGET_FS)
    except Exception as exc:
        window.peak_error = type(exc).__name__ + ": " + str(exc)[:160]
        window.peaks = np.empty(0, dtype=np.int64)
    if "hrhrv" in input_kinds:
        window.hrhrv = hr_hrv_ms(window.peaks)
    if "beats" in input_kinds and not window.peak_error:
        try:
            window.beats, window.rr, window.n_beats = stack_beats(
                window.trace, peaks=window.peaks
            )
        except Exception as exc:
            window.beat_error = type(exc).__name__ + ": " + str(exc)[:160]
    return window
