from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ReconstructionLoss(nn.Module):
    def __init__(self, mode: str = "mse") -> None:
        super().__init__()
        valid_modes = {"mse", "l1", "smooth_l1"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}.")
        self.mode = mode

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "mse":
            return F.mse_loss(prediction, target)
        if self.mode == "l1":
            return F.l1_loss(prediction, target)
        return F.smooth_l1_loss(prediction, target)
