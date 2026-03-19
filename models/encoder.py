from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.decoder import ReconstructionDecoder
from models.inception import InceptionEncoder
from models.transformer import TemporalTransformerEncoder


@dataclass
class ECGEncoderConfig:
    input_channels: int = 12
    inception_depth: int = 6
    inception_out_channels: int = 32
    inception_kernel_sizes: Sequence[int] = (39, 19, 9)
    bottleneck_channels: int = 32
    transformer_dim: int = 128
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_feedforward_dim: int = 256
    local_pool_bins: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 5000


@dataclass
class EncoderOutputs:
    cnn_features: torch.Tensor
    local_embedding: torch.Tensor
    transformer_tokens: torch.Tensor
    global_embedding: torch.Tensor
    reconstruction: torch.Tensor


class ECGContrastiveAutoencoder(nn.Module):
    def __init__(self, config: ECGEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.cnn = InceptionEncoder(
            input_channels=config.input_channels,
            depth=config.inception_depth,
            out_channels=config.inception_out_channels,
            kernel_sizes=config.inception_kernel_sizes,
            bottleneck_channels=config.bottleneck_channels,
            use_residual=True,
        )
        self.local_pool = nn.AdaptiveAvgPool1d(config.local_pool_bins)
        self.transformer = TemporalTransformerEncoder(
            input_dim=self.cnn.output_channels,
            model_dim=config.transformer_dim,
            num_layers=config.transformer_layers,
            num_heads=config.transformer_heads,
            feedforward_dim=config.transformer_feedforward_dim,
            dropout=config.dropout,
            max_length=config.max_sequence_length,
        )
        self.global_normalization = nn.LayerNorm(config.transformer_dim)
        self.decoder = ReconstructionDecoder(
            input_dim=config.transformer_dim,
            output_channels=config.input_channels,
            hidden_dim=config.transformer_dim,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> EncoderOutputs:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [batch, leads, time], got {tuple(x.shape)}.")
        cnn_features = self.cnn(x)
        local_embedding = torch.flatten(self.local_pool(cnn_features), start_dim=1)
        transformer_tokens = self.transformer(cnn_features.transpose(1, 2))
        global_embedding = self.global_normalization(transformer_tokens.mean(dim=1))
        reconstruction = self.decoder(transformer_tokens)
        return EncoderOutputs(
            cnn_features=cnn_features,
            local_embedding=local_embedding,
            transformer_tokens=transformer_tokens,
            global_embedding=global_embedding,
            reconstruction=reconstruction,
        )

    def embed(self, x: torch.Tensor, embedding_type: str = "global", normalize: bool = True) -> torch.Tensor:
        outputs = self.forward(x)
        if embedding_type in {"retrieval", "global"}:
            embedding = outputs.global_embedding
        elif embedding_type == "local":
            embedding = outputs.local_embedding
        else:
            raise ValueError("embedding_type must be one of {'retrieval', 'global', 'local'}.")
        return embedding if not normalize else F.normalize(embedding, dim=-1)
