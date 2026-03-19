from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from losses.contrastive import NTXentLoss
from losses.reconstruction import ReconstructionLoss
from models.encoder import EncoderOutputs


@dataclass
class LossWeights:
    local: float = 1.0
    global_: float = 1.0
    reconstruction: float = 1.0


class ECGTrainingObjective(nn.Module):
    def __init__(
        self,
        local_temperature: float = 0.1,
        global_temperature: float = 0.1,
        reconstruction_mode: str = "mse",
        weights: LossWeights | None = None,
    ) -> None:
        super().__init__()
        self.local_contrastive = NTXentLoss(temperature=local_temperature)
        self.global_contrastive = NTXentLoss(temperature=global_temperature)
        self.reconstruction_loss = ReconstructionLoss(mode=reconstruction_mode)
        self.weights = weights or LossWeights()

    def forward(
        self,
        outputs_view1: EncoderOutputs,
        outputs_view2: EncoderOutputs,
        target_view1: torch.Tensor,
        target_view2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local_loss = self.local_contrastive(outputs_view1.local_embedding, outputs_view2.local_embedding)
        global_loss = self.global_contrastive(outputs_view1.projection, outputs_view2.projection)
        reconstruction_loss_view1 = self.reconstruction_loss(outputs_view1.reconstruction, target_view1)
        reconstruction_loss_view2 = self.reconstruction_loss(outputs_view2.reconstruction, target_view2)
        reconstruction_loss = 0.5 * (reconstruction_loss_view1 + reconstruction_loss_view2)

        total_loss = (
            self.weights.local * local_loss
            + self.weights.global_ * global_loss
            + self.weights.reconstruction * reconstruction_loss
        )
        return {
            "loss": total_loss,
            "local_loss": local_loss,
            "global_loss": global_loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_loss_view1": reconstruction_loss_view1,
            "reconstruction_loss_view2": reconstruction_loss_view2,
        }
