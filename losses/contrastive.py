from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape[0] != z2.shape[0]:
            raise ValueError("Both views must have the same batch size for contrastive learning.")
        if z1.ndim > 2:
            z1 = torch.flatten(z1, start_dim=1)
        if z2.ndim > 2:
            z2 = torch.flatten(z2, start_dim=1)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.size(0)
        embeddings = torch.cat([z1, z2], dim=0)
        logits = embeddings @ embeddings.T / self.temperature
        logits = logits.masked_fill(torch.eye(2 * batch_size, device=logits.device, dtype=torch.bool), -1e9)

        targets = torch.arange(batch_size, device=logits.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)
        return F.cross_entropy(logits, targets)
