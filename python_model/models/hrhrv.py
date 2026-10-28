"""PyTorch architecture for the linear HR+HRV baseline.

Normalize mean RR and SDNN features, then produce one AF/AFL logit.

Use ``runtime.load_model`` to construct a model with its trained weights.
"""

import torch
from torch import nn


class HRHRV(nn.Module):
    """Predict AF or AFL from finite mean RR and SDNN features.

    Construction initializes weights; ``runtime.load_model`` loads the trained
    checkpoint, including feature normalization buffers.
    """

    input_kind = "hrhrv"

    def __init__(self):
        """Initialize the linear layer and feature normalization buffers."""
        super().__init__()
        # Inputs (in milliseconds)
        # - mean RR inteval duration (HR proxy)
        # - SDNN, standard deviation of corrected RR intervals (HRV)
        #
        # hr_mean = [mean(mean RR), mean(SDNN)]
        # hr_std  = [std(mean RR), std(SDNN)]
        self.register_buffer("hr_mean", torch.zeros(2, dtype=torch.float32))
        self.register_buffer("hr_std", torch.ones(2, dtype=torch.float32))
        self.fc = nn.Sequential(nn.Linear(2, 1))

    def forward(self, x):
        """Compute AF/AFL logits from RR summary features.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 2), dtype torch.float32
            Mean RR and sample SDNN, in that order in milliseconds.
            Both features must be finite.

        Returns
        -------
        logits : torch.Tensor, shape (B, 1)
            One score per window, before sigmoid conversion.

        Raises
        ------
        ValueError
            If the input shape is not (B, 2) or either feature is nonfinite.
        """
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("expected HR/HRV shape (B, 2), in milliseconds")
        if not torch.isfinite(x).all():
            raise ValueError("HRHRV requires finite mean RR and SDNN features")
        return self.fc((x - self.hr_mean) / (self.hr_std + 1e-8))
