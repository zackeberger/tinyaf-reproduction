"""PyTorch architecture for the SmallNN (Kim) baseline.

Separable convolution blocks encode a conditioned ECG window and pool its
features to produce one AF/AFL logit.

Use ``runtime.load_model`` to construct a model with its trained weights.
"""

from torch import nn
from torch.nn import functional as F


class SeparableExpandIRBlock1D(nn.Module):
    """Apply depthwise convolution, sixfold expansion, and linear projection.

    Parameters
    ----------
    in_ch, out_ch : int
        Input and output channel counts.
    kernel_size, stride : int
        Depthwise convolution kernel width and stride, in samples.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride):
        """Initialize convolutions, normalization, and the optional residual path."""
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size, stride=stride,
                            padding=kernel_size // 2, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm1d(in_ch)
        self.pw_expand = nn.Conv1d(in_ch, in_ch * 6, 1, bias=False)
        self.pw_expand_bn = nn.BatchNorm1d(in_ch * 6)
        self.pw_project = nn.Conv1d(in_ch * 6, out_ch, 1, bias=False)
        self.pw_project_bn = nn.BatchNorm1d(out_ch)
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        """Encode features, adding the input when channel counts and stride allow.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_ch, L)
            Input feature sequences.

        Returns
        -------
        torch.Tensor, shape (B, out_ch, L_out)
            Projected features at the convolution's output length.
        """
        identity = x
        x = F.relu(self.dw_bn(self.dw(x)), inplace=True)
        x = F.relu(self.pw_expand_bn(self.pw_expand(x)), inplace=True)
        x = self.pw_project_bn(self.pw_project(x))
        return x + identity if self.use_residual else x


class SmallNN(nn.Module):
    """Encode a conditioned ECG with separable convolutions to predict AF or AFL.

    Construction initializes weights; ``runtime.load_model`` loads a trained
    checkpoint.
    """

    input_kind = "trace"

    def __init__(self):
        """Initialize convolution blocks, pooling, and the output layer."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 8, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(8), nn.ReLU(inplace=True),
        )
        self.block1 = SeparableExpandIRBlock1D(8, 8, 7, 1)
        self.dropout = nn.Dropout(0.2)
        self.block2 = SeparableExpandIRBlock1D(8, 12, 7, 2)
        self.block3 = SeparableExpandIRBlock1D(12, 16, 5, 2)
        self.block4 = SeparableExpandIRBlock1D(16, 24, 5, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(24, 1)

    def forward(self, x):
        """Compute AF/AFL logits from conditioned ECG windows.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1000), dtype torch.float32
            Filtered and normalized 10-second, single-lead ECGs at 100 Hz.

        Returns
        -------
        logits : torch.Tensor, shape (B, 1)
            One score per window, before sigmoid conversion.

        Raises
        ------
        ValueError
            If the input shape is not (B, 1000).
        """
        if x.ndim != 2 or x.shape[1] != 1000:
            raise ValueError("expected conditioned ECG shape (B, 1000)")
        x = x.view(x.shape[0], 1, 1000)
        x = self.dropout(self.block1(self.stem(x)))
        x = self.block4(self.block3(self.block2(x)))
        return self.fc(self.gap(x).squeeze(-1))
