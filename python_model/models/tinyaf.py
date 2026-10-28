"""PyTorch architectures for TinyAF-S, TinyAF-M, and TinyAF-L.

Each model encodes beat waveforms, combines each beat's features with its
RR interval, and pools across valid beats to produce one AF/AFL logit.

Use ``runtime.load_model`` to construct a model with its trained weights.
"""

import torch
from torch import nn
from torch.nn import functional as F

# Input geometry and pooling configuration.
N_BEATS = 20
BEAT_W = 100  # Samples per resampled beat
POOL_BINS = 8
BEAT_KERNEL = 3
BEAT_STRIDE = 1

# Fixed RR normalization constants, in seconds.
RR_MEAN, RR_STD = 0.783, 0.210

# kernel1, stride1, filters1, kernel2, stride2, filters2, embed, beat_ch, hidden
RUNGS = {
    "S": (8, 6, 4, 2, 2, 8, 6, 6, 12),
    "M": (8, 6, 4, 2, 2, 16, 10, 10, 20),
    "L": (8, 6, 8, 2, 2, 16, 16, 16, 32),
}

# Expected number of trainable parameters for each model size.
RUNG_PARAMS = {"S": 1279, "M": 3611, "L": 7657}


def _out_len(L, k, s):
    """Calculate a Conv1d output length with padding ``k // 2``.

    Parameters
    ----------
    L : int
        Input length in samples.
    k : int
        Kernel width in samples.
    s : int
        Convolution stride in samples.

    Returns
    -------
    int
        Output length in samples.
    """
    return (L + 2 * (k // 2) - k) // s + 1


class TinyAF(nn.Module):
    """Encode 20 beat waveforms and their RR intervals to predict AF or AFL.

    Use ``rung`` to select the S, M, or L architecture. Construction
    initializes weights; ``runtime.load_model`` loads a trained checkpoint.

    Parameters
    ----------
    kernel1, kernel2 : int
        Kernel widths for the two convolutions within each beat.
    stride1, stride2 : int
        Strides for those convolutions.
    filters1, filters2 : int
        Output channels for those convolutions.
    embed : int
        Number of learned features per beat.
    beat_ch : int
        Output channels of the convolution across beats.
    hidden : int
        Width of the final hidden layer.
    """

    input_kind = "beats"

    def __init__(self, kernel1, stride1, filters1, kernel2, stride2, filters2,
                 embed, beat_ch, hidden):
        """Initialize network layers and normalization buffers."""
        super().__init__()

        self.n_beats = N_BEATS
        self.beat_w = BEAT_W
        self.embed = embed
        self.beat_ch = beat_ch

        self.conv1 = nn.Conv1d(1, filters1, kernel1, stride=stride1,
                               padding=kernel1 // 2)
        self.conv2 = nn.Conv1d(filters1, filters2, kernel2, stride=stride2,
                               padding=kernel2 // 2)
        L1 = _out_len(BEAT_W, kernel1, stride1)
        L2 = _out_len(L1, kernel2, stride2)
        assert L1 > 0 and L2 > 0
        self.per_beat = filters2 * L2
        self.beat_enc = nn.Linear(self.per_beat, embed)

        # Add one RR feature to each beat embedding.
        self.beat_conv = nn.Conv1d(embed + 1, beat_ch, BEAT_KERNEL,
                                   stride=BEAT_STRIDE, padding=BEAT_KERNEL // 2)
        n_out = _out_len(N_BEATS, BEAT_KERNEL, BEAT_STRIDE)
        assert n_out == N_BEATS, (
            f"stride {BEAT_STRIDE} must leave the beat axis at {N_BEATS}; "
            f"got {n_out}. The mask simplification below depends on it."
        )

        self.fc1 = nn.Linear(beat_ch * POOL_BINS, hidden)
        self.fc = nn.Linear(hidden, 1)

        self.register_buffer("rr_mean", torch.tensor(RR_MEAN, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("rr_std", torch.tensor(RR_STD, dtype=torch.float32),
                             persistent=False)

    @classmethod
    def rung(cls, name):
        """Construct an S, M, or L network with randomly initialized weights.

        Parameters
        ----------
        name : str
            Model size: ``"S"``, ``"M"``, or ``"L"`` (case-insensitive).

        Returns
        -------
        TinyAF
            Network with the selected architecture.

        Raises
        ------
        ValueError
            If the model size is unknown.
        """
        key = str(name).upper()
        if key not in RUNGS:
            raise ValueError(f"unknown rung {name!r}; have {sorted(RUNGS)}")
        net = cls(*RUNGS[key])
        n = sum(p.numel() for p in net.parameters())
        assert n == RUNG_PARAMS[key], (
            f"TinyAF-{key} built with {n:,} parameters, frozen table says "
            f"{RUNG_PARAMS[key]:,}"
        )
        return net

    def forward(self, beats, rr_seconds):
        """Compute AF/AFL logits from beat waveforms and their RR intervals.

        Parameters
        ----------
        beats : torch.Tensor, shape (B, 20, 100), dtype torch.float32
            Conditioned beat waveforms, with unused beat slots zero-padded.
        rr_seconds : torch.Tensor, shape (B, 20), dtype torch.float32
            Corresponding RR intervals in seconds. Zero marks an unused slot.

        Returns
        -------
        logits : torch.Tensor, shape (B, 1)
            One score per window, before sigmoid conversion.

        Raises
        ------
        ValueError
            If the input shapes do not match the model's beat geometry.
        """
        if beats.ndim != 3 or beats.shape[1:] != (self.n_beats, self.beat_w):
            raise ValueError(f"expected beats shape (B, {self.n_beats}, {self.beat_w})")
        B = beats.shape[0]
        if rr_seconds.shape != (B, self.n_beats):
            raise ValueError(f"expected RR shape ({B}, {self.n_beats})")

        # Encode each beat with shared weights.
        h = beats.reshape(B * self.n_beats, 1, self.beat_w)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = h.reshape(B * self.n_beats, self.per_beat)
        h = F.relu(self.beat_enc(h))
        h = h.reshape(B, self.n_beats, self.embed)

        # Combine waveform features with RR, then convolve across beats.
        rr = (rr_seconds - self.rr_mean) / (self.rr_std + 1e-8)
        h = torch.cat([h, rr.unsqueeze(-1)], dim=-1)
        h = F.relu(self.beat_conv(h.transpose(1, 2)))

        # Exclude padded beat slots from each pooled average.
        live = (rr_seconds != 0).to(h.dtype).unsqueeze(1)
        h = h * live
        num = F.adaptive_avg_pool1d(h, POOL_BINS)
        den = F.adaptive_avg_pool1d(live, POOL_BINS)
        # Empty bins remain zero without division by zero.
        h = num / den.clamp_min(1e-6)
        h = h.reshape(B, self.beat_ch * POOL_BINS)

        h = F.relu(self.fc1(h))
        return self.fc(h)
