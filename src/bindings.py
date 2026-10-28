"""ctypes access to caller-owned TinyAF workspaces."""

from pathlib import Path
import ctypes as ct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
F64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
U16 = np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags="C_CONTIGUOUS")
I32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")


class Result(ct.Structure):
    """C prediction fields: input length, rate in Hz, beat count, and AF/AFL scores.

    On success, ``ok`` is 1 and ``prediction`` is 0 or 1. On failure, ``ok`` is 0,
    ``prediction`` is -1, and the logit and probability are NaN.
    """

    _fields_ = [
        ("ok", ct.c_int),
        ("raw_len", ct.c_int),
        ("sampling_rate", ct.c_int),
        ("n_beats", ct.c_int),
        ("logit", ct.c_float),
        ("probability", ct.c_float),
        ("threshold", ct.c_float),
        ("prediction", ct.c_int),
    ]


class Predictor:
    """Call one compiled C model using a reusable workspace owned by this object.

    Parameters
    ----------
    rung : str, optional
        Model size: ``"S"``, ``"M"``, or ``"L"``. Default is ``"S"``.
    mode : str, optional
        C implementation: ``"reference"`` or ``"optimized"``.
        Default is ``"optimized"``.

    Notes
    -----
    Loads ``compile/<mode>/<rung>/libtinyaf.so`` with native-input and diagnostic
    APIs enabled. Does not build it. ``size`` is the workspace size in bytes.
    Calls on the same instance must be serialized because they share storage.
    """

    def __init__(self, rung="S", mode="optimized"):
        """Load the selected library, allocate its workspace, and bind C functions."""
        if rung not in {"S", "M", "L"} or mode not in {"reference", "optimized"}:
            raise ValueError("unknown TinyAF rung or implementation")
        self.lib = ct.CDLL(str(ROOT / "compile" / mode / rung / "libtinyaf.so"))
        self.lib.tinyaf_workspace_size.restype = ct.c_size_t
        self.size = self.lib.tinyaf_workspace_size()
        # ctypes allocates suitably aligned storage; the C workspace layout is opaque.
        self.workspace = ct.create_string_buffer(self.size)
        for name, arg in [("tinyaf_predict_native", F64), ("tinyaf_predict_adc", U16)]:
            fn = getattr(self.lib, name)
            fn.argtypes = (
                [arg, ct.c_int]
                + ([ct.c_int] if name.endswith("native") else [])
                + [ct.c_float, ct.POINTER(Result), ct.c_void_p]
            )
            fn.restype = ct.c_int
        self.lib.tinyaf_debug_native.argtypes = [
            F64,
            ct.c_int,
            ct.c_int,
            F32,
            F32,
            F32,
            I32,
            ct.c_void_p,
        ]
        self.lib.tinyaf_debug_correct.argtypes = [I32, ct.c_int, ct.c_void_p]

    def native(self, raw, fs, threshold=-1):
        """Predict from one physical ECG window at its original sampling rate.

        Parameters
        ----------
        raw : array_like, shape (10 * fs,)
            Ten seconds of single-lead ECG samples in physical units.
            Converted to contiguous float64.
        fs : int
            Input sampling frequency in Hz; must be supported by the C build.
        threshold : float, optional
            Classification threshold in [0, 1], or -1 for the compiled default.

        Returns
        -------
        status : int
            Zero on success; a negative status code from ``tinyaf.h`` on failure.
        result : Result
            Prediction fields. Logit and probability are NaN on failure.
        """
        raw = np.ascontiguousarray(raw, dtype=np.float64)
        out = Result()
        status = self.lib.tinyaf_predict_native(
            raw, len(raw), fs, threshold, ct.byref(out), self.workspace
        )
        return status, out

    def adc(self, raw, threshold=-1):
        """Predict from one Quanta ADC window.

        Parameters
        ----------
        raw : array_like, shape (2560,)
            Ten seconds of 256 Hz integer ADC counts in 0..4095.
            Converted to contiguous uint16 before calling C.
        threshold : float, optional
            Classification threshold in [0, 1], or -1 for the compiled default.

        Returns
        -------
        status : int
            Zero on success; a negative status code from ``tinyaf.h`` on failure.
        result : Result
            Prediction fields. Logit and probability are NaN on failure.
        """
        raw = np.ascontiguousarray(raw, dtype=np.uint16)
        out = Result()
        status = self.lib.tinyaf_predict_adc(
            raw, len(raw), threshold, ct.byref(out), self.workspace
        )
        return status, out

    def debug(self, raw, fs):
        """Return C preprocessing stages for one physical ECG window.

        Parameters
        ----------
        raw : array_like, shape (10 * fs,)
            Ten seconds of single-lead ECG samples in physical units.
            Converted to contiguous float64.
        fs : int
            Input sampling frequency in Hz; must be supported by the C build.

        Returns
        -------
        n : int
            Number of corrected peaks, or a negative C error code.
        trace : numpy.ndarray, shape (1000,), dtype float32
            Conditioned ECG at 100 Hz.
        beats : numpy.ndarray, shape (20, 100), dtype float32
            Resampled beat waveforms. Unused rows are zero-padded.
        rr : numpy.ndarray, shape (20,), dtype float32
            RR lengths in samples at 100 Hz. Unused entries are zero.
        peaks : numpy.ndarray, shape (max(n, 0),), dtype int32
            Corrected peak indices into ``trace``. Empty on failure.

        Notes
        -----
        On failure, intermediate arrays may be only partially populated.
        """
        raw = np.ascontiguousarray(raw, dtype=np.float64)
        trace = np.zeros(1000, np.float32)
        beats = np.zeros(2000, np.float32)
        rr = np.zeros(20, np.float32)
        peaks = np.zeros(128, np.int32)
        n = self.lib.tinyaf_debug_native(
            raw, len(raw), fs, trace, beats, rr, peaks, self.workspace
        )
        return n, trace, beats.reshape(20, 100), rr, peaks[: max(n, 0)]
