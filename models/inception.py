from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _make_odd(kernel_size: int) -> int:
    return kernel_size if kernel_size % 2 == 1 else kernel_size - 1


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (39, 19, 9),
        bottleneck_channels: int = 32,
        use_bottleneck: bool = True,
    ) -> None:
        super().__init__()
        normalized_kernels = tuple(_make_odd(kernel_size) for kernel_size in kernel_sizes)
        branch_in_channels = bottleneck_channels if use_bottleneck and in_channels > 1 else in_channels
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if branch_in_channels != in_channels
            else nn.Identity()
        )
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    branch_in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
                for kernel_size in normalized_kernels
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.normalization = nn.BatchNorm1d(out_channels * (len(normalized_kernels) + 1))
        self.activation = nn.GELU()
        self.output_channels = out_channels * (len(normalized_kernels) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_input = self.bottleneck(x)
        outputs = [branch(bottleneck_input) for branch in self.branches]
        outputs.append(self.pool_branch(x))
        x = torch.cat(outputs, dim=1)
        x = self.normalization(x)
        return self.activation(x)


class ResidualShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.projection(residual))


class InceptionEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        depth: int = 6,
        out_channels: int = 32,
        kernel_sizes: Sequence[int] = (39, 19, 9),
        bottleneck_channels: int = 32,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        current_channels = input_channels
        residual_channels = input_channels
        for index in range(depth):
            block = InceptionModule(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_bottleneck=True,
            )
            self.blocks.append(block)
            current_channels = block.output_channels
            if self.use_residual and index % 3 == 2:
                self.shortcuts.append(ResidualShortcut(residual_channels, current_channels))
                residual_channels = current_channels

        self.output_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        shortcut_index = 0
        for index, block in enumerate(self.blocks):
            x = block(x)
            if self.use_residual and index % 3 == 2:
                x = self.shortcuts[shortcut_index](x, residual)
                residual = x
                shortcut_index += 1
        return x
