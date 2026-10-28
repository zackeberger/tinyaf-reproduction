"""PyTorch architecture for the LargeNN (Ribeiro) baseline.

Residual convolution blocks encode a conditioned ECG window and pool its
features to produce one AF/AFL logit.

Use ``runtime.load_model`` to construct a model with its trained weights.
"""

import math

from torch import nn
from torch.nn import functional as F


class Conv1dSame(nn.Module):
    """Convolve with SAME padding to produce length ``ceil(L / stride)``.

    Parameters
    ----------
    in_channels, out_channels : int
        Input and output channel counts.
    kernel_size : int
        Convolution kernel width in samples.
    stride : int, optional
        Convolution stride in samples. Default is 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        """Initialize the convolution and store its padding geometry."""
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=False)

    def forward(self, x):
        """Pad the sequence and apply the convolution.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_channels, L)
            Input feature sequences.

        Returns
        -------
        torch.Tensor, shape (B, out_channels, ceil(L / stride))
            Convolved features, with any extra padding placed on the right.
        """
        length = x.size(-1)
        out_length = math.ceil(length / self.stride)
        padding = max((out_length - 1) * self.stride + self.kernel_size - length, 0)
        if padding:
            x = F.pad(x, (padding // 2, padding - padding // 2))
        return self.conv(x)


class ResidualUnit1DFixed(nn.Module):
    """Combine convolution and skip paths with kernel 16, stride 4, dropout 0.2.

    Parameters
    ----------
    in_channels, out_channels : int
        Input and output channel counts for both paths.
    """

    def __init__(self, in_channels, out_channels):
        """Initialize convolutions, normalization, dropout, and the skip path."""
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = Conv1dSame(in_channels, out_channels, 16)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(1.0 - 0.8)
        self.conv2 = Conv1dSame(out_channels, out_channels, 16, stride=4)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(1.0 - 0.8)
        self.skip_pool = nn.MaxPool1d(4, stride=4, ceil_mode=True)
        self.skip_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x, y):
        """Downsample and combine the main and skip feature sequences.

        Parameters
        ----------
        x, y : torch.Tensor, shape (B, in_channels, L)
            Main-path and skip-path inputs, respectively.

        Returns
        -------
        x : torch.Tensor, shape (B, out_channels, ceil(L / 4))
            Features after normalization, activation, and dropout.
        y_out : torch.Tensor, shape (B, out_channels, ceil(L / 4))
            Features after residual addition, for the next skip path.
        """
        y_skip = self.skip_conv(self.skip_pool(y))
        x = self.drop1(self.act(self.bn1(self.conv1(x))))
        x = self.conv2(x) + y_skip
        y_out = x
        x = self.drop2(self.act(self.bn2(x)))
        return x, y_out


class LargeNN(nn.Module):
    """Encode a conditioned ECG with residual convolutions to predict AF or AFL.

    Construction initializes weights; ``runtime.load_model`` loads a trained
    checkpoint.
    """

    input_kind = "trace"

    def __init__(self):
        """Initialize the stem, residual units, pooling, and output layer."""
        super().__init__()
        self.stem_conv = Conv1dSame(1, 64, 16)
        self.stem_bn = nn.BatchNorm1d(64)
        self.stem_act = nn.ReLU(inplace=True)
        self.ru1 = ResidualUnit1DFixed(64, 128)
        self.ru2 = ResidualUnit1DFixed(128, 196)
        self.ru3 = ResidualUnit1DFixed(196, 256)
        self.ru4 = ResidualUnit1DFixed(256, 320)
        self.pool_to_16 = nn.AdaptiveAvgPool1d(16)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 320, 1)

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
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x, y = self.ru1(x, x)
        x, y = self.ru2(x, y)
        x, y = self.ru3(x, y)
        x, _ = self.ru4(x, y)
        return self.fc(self.flatten(self.pool_to_16(x)))
