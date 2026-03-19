from __future__ import annotations

from dataclasses import dataclass

import torch

from data.augmentations import TwoViewECGAugmentor
from data.dataset import ECGDataConfig, build_split_dataloaders
from losses.total_loss import ECGTrainingObjective, LossWeights
from models.encoder import ECGContrastiveAutoencoder, ECGEncoderConfig
from training.trainer import ContrastiveAutoencoderTrainer


@dataclass
class TrainConfig:
    data_root: str | None = None
    batch_size: int = 8
    input_channels: int = 12
    sequence_length: int = 2500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    epochs: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    steps: int = 1


def build_trainer(
    model_config: ECGEncoderConfig | None = None,
    loss_weights: LossWeights | None = None,
    device: str = "cpu",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> ContrastiveAutoencoderTrainer:
    model_config = model_config or ECGEncoderConfig()
    model = ECGContrastiveAutoencoder(model_config)
    objective = ECGTrainingObjective(weights=loss_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    augmentor = TwoViewECGAugmentor()
    return ContrastiveAutoencoderTrainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        augmentor=augmentor,
        device=device,
    )


def smoke_test(config: TrainConfig | None = None) -> list[dict[str, float]]:
    config = config or TrainConfig()
    trainer = build_trainer(
        model_config=ECGEncoderConfig(input_channels=config.input_channels, max_sequence_length=config.sequence_length),
        device=config.device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    metrics_history: list[dict[str, float]] = []
    for _ in range(config.steps):
        batch = torch.randn(config.batch_size, config.input_channels, config.sequence_length)
        metrics_history.append(trainer.step(batch, train=True))
    return metrics_history


def train_with_dataloaders(config: TrainConfig | None = None) -> dict[str, object]:
    config = config or TrainConfig()
    if config.data_root is None:
        raise ValueError("config.data_root must be set to train on real ECG files.")

    trainer = build_trainer(
        model_config=ECGEncoderConfig(
            input_channels=config.input_channels,
            max_sequence_length=config.sequence_length,
        ),
        device=config.device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    dataloaders = build_split_dataloaders(
        ECGDataConfig(
            data_root=config.data_root,
            batch_size=config.batch_size,
            eval_batch_size=config.batch_size,
            num_leads=config.input_channels,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    )

    history: list[dict[str, object]] = []
    for epoch in range(config.epochs):
        train_metrics = trainer.fit_epoch(dataloaders["train"])
        epoch_result: dict[str, object] = {
            "epoch": epoch + 1,
            "train": train_metrics,
        }
        if "val" in dataloaders:
            epoch_result["val"] = trainer.evaluate_epoch(dataloaders["val"])
        history.append(epoch_result)

    results: dict[str, object] = {
        "history": history,
        "available_splits": sorted(dataloaders.keys()),
    }
    if "test" in dataloaders:
        results["test"] = trainer.evaluate_epoch(dataloaders["test"])
    return results
