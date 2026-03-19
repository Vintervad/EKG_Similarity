from __future__ import annotations

import torch
from torch import nn


class ReconstructionDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_channels: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.decoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, output_channels, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.transpose(1, 2)
        return self.decoder(x)
