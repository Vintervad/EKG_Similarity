from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_length: int = 5000) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_length = max_length
        self.register_buffer("encoding", self._build_encoding(max_length, model_dim), persistent=False)

    @staticmethod
    def _build_encoding(length: int, model_dim: int) -> torch.Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / model_dim))
        encoding = torch.zeros(length, model_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        return encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len <= self.encoding.size(1):
            positional_encoding = self.encoding[:, :seq_len]
        else:
            positional_encoding = self._build_encoding(seq_len, self.model_dim).to(device=x.device)
        return x + positional_encoding.to(device=x.device, dtype=x.dtype)


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        max_length: int = 5000,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.input_projection = nn.Identity() if input_dim == model_dim else nn.Linear(input_dim, model_dim)
        self.position_encoding = SinusoidalPositionalEncoding(model_dim=model_dim, max_length=max_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_normalization = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        return self.output_normalization(x)
