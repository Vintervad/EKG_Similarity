from __future__ import annotations

import torch
import torch.nn.functional as F
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
        self.view_embedding = nn.Embedding(2, input_dim)
        self.decoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, output_channels, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, view_id: int | None = None) -> torch.Tensor:
        if view_id is not None:
            view_ids = torch.full(
                (tokens.size(0),),
                view_id,
                device=tokens.device,
                dtype=torch.long,
            )
            tokens = tokens + self.view_embedding(view_ids).to(dtype=tokens.dtype).unsqueeze(1)
        x = tokens.transpose(1, 2)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        return self.decoder(x)
