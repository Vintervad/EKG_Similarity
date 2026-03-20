from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    checkpoint_dir: str = "checkpoints"
    save_every_batch: bool = True
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.0


def _checkpoint_dir(path: str | Path) -> Path:
    checkpoint_dir = Path(path)
    return checkpoint_dir if checkpoint_dir.is_absolute() else Path.cwd() / checkpoint_dir


def _is_improvement(current_metric: float, best_metric: float, min_delta: float) -> bool:
    return current_metric < (best_metric - min_delta)


def _normalize_patience(patience: int | None) -> int | None:
    if patience is None:
        return None
    return None if patience < 0 else patience


def _metric_fieldnames() -> list[str]:
    return [
        "split",
        "epoch",
        "global_step",
        "batch_index",
        "loss",
        "local_loss",
        "global_loss",
        "reconstruction_loss",
        "reconstruction_loss_view1",
        "reconstruction_loss_view2",
        "batch_size",
    ]


def _epoch_metric_fieldnames() -> list[str]:
    return [
        "epoch",
        "global_step",
        "selection_metric",
        "train_loss",
        "train_local_loss",
        "train_global_loss",
        "train_reconstruction_loss",
        "train_reconstruction_loss_view1",
        "train_reconstruction_loss_view2",
        "train_batch_size",
        "val_loss",
        "val_local_loss",
        "val_global_loss",
        "val_reconstruction_loss",
        "val_reconstruction_loss_view1",
        "val_reconstruction_loss_view2",
        "val_batch_size",
    ]


def _initialize_metrics_file(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _append_metrics_row(
    path: Path,
    fieldnames: list[str],
    *,
    split: str,
    epoch: int,
    global_step: int,
    batch_index: int,
    metrics: dict[str, float],
) -> None:
    row = {
        "split": split,
        "epoch": epoch,
        "global_step": global_step,
        "batch_index": batch_index,
    }
    for field in fieldnames:
        if field in row:
            continue
        row[field] = float(metrics.get(field, 0.0))
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def _append_epoch_metrics_row(
    path: Path,
    fieldnames: list[str],
    *,
    epoch: int,
    global_step: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
    selection_metric: float,
) -> None:
    row: dict[str, float | int | str] = {
        "epoch": epoch,
        "global_step": global_step,
        "selection_metric": selection_metric,
    }
    for key, value in train_metrics.items():
        row[f"train_{key}"] = float(value)
    for field in fieldnames:
        row.setdefault(field, "")
    if val_metrics is not None:
        for key, value in val_metrics.items():
            row[f"val_{key}"] = float(value)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def _evaluate_split(
    trainer: ContrastiveAutoencoderTrainer,
    data_loader: Any,
    *,
    split: str,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    for batch in data_loader:
        metrics = trainer.step(batch, train=False)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    if count == 0:
        raise ValueError(f"{split} dataloader is empty.")
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(
    trainer: ContrastiveAutoencoderTrainer,
    checkpoint_path: str | Path,
    *,
    epoch: int,
    global_step: int,
    batch_index: int | None,
    metrics: dict[str, float] | None = None,
    best_metric: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "batch_index": batch_index,
        "metrics": metrics or {},
    }
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, checkpoint_path)
    return checkpoint_path


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
    patience = _normalize_patience(config.early_stopping_patience)

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

    checkpoint_dir = _checkpoint_dir(config.checkpoint_dir)
    batches_dir = checkpoint_dir / "batches"
    metrics_dir = checkpoint_dir / "metrics"
    batch_fieldnames = _metric_fieldnames()
    epoch_fieldnames = _epoch_metric_fieldnames()
    train_metrics_path = metrics_dir / "train_batch_metrics.csv"
    epoch_metrics_path = metrics_dir / "epoch_metrics.csv"
    stale_val_batch_metrics_path = metrics_dir / "val_batch_metrics.csv"
    _initialize_metrics_file(train_metrics_path, batch_fieldnames)
    _initialize_metrics_file(epoch_metrics_path, epoch_fieldnames)
    if stale_val_batch_metrics_path.exists():
        stale_val_batch_metrics_path.unlink()

    history: list[dict[str, object]] = []
    global_step = 0
    best_metric = float("inf")
    best_checkpoint_path: str | None = None
    selection_metric_name = "val_loss" if "val" in dataloaders else "train_loss"
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch: int | None = None

    for epoch in range(config.epochs):
        trainer.model.train(True)
        epoch_totals: dict[str, float] = {}
        batch_count = 0

        for batch_index, batch in enumerate(dataloaders["train"], start=1):
            metrics = trainer.step(batch, train=True)
            _append_metrics_row(
                train_metrics_path,
                batch_fieldnames,
                split="train",
                epoch=epoch + 1,
                global_step=global_step + 1,
                batch_index=batch_index,
                metrics=metrics,
            )
            for key, value in metrics.items():
                epoch_totals[key] = epoch_totals.get(key, 0.0) + value
            batch_count += 1
            global_step += 1

            if config.save_every_batch:
                batch_checkpoint = batches_dir / f"epoch_{epoch + 1:04d}_batch_{batch_index:06d}.pt"
                save_checkpoint(
                    trainer,
                    batch_checkpoint,
                    epoch=epoch + 1,
                    global_step=global_step,
                    batch_index=batch_index,
                    metrics=metrics,
                    metadata={"selection_metric_name": selection_metric_name},
                )
                save_checkpoint(
                    trainer,
                    checkpoint_dir / "latest.pt",
                    epoch=epoch + 1,
                    global_step=global_step,
                    batch_index=batch_index,
                    metrics=metrics,
                    best_metric=best_metric if best_metric != float("inf") else None,
                    metadata={"selection_metric_name": selection_metric_name},
                )

        if batch_count == 0:
            raise ValueError("Train dataloader is empty.")
        train_metrics = {key: value / batch_count for key, value in epoch_totals.items()}
        epoch_result: dict[str, object] = {
            "epoch": epoch + 1,
            "train": train_metrics,
        }
        if "val" in dataloaders:
            epoch_result["val"] = _evaluate_split(
                trainer,
                dataloaders["val"],
                split="val",
            )
        selection_metrics = epoch_result["val"] if "val" in epoch_result else epoch_result["train"]
        selection_metric = float(selection_metrics["loss"])
        _append_epoch_metrics_row(
            epoch_metrics_path,
            epoch_fieldnames,
            epoch=epoch + 1,
            global_step=global_step,
            train_metrics=train_metrics,
            val_metrics=epoch_result.get("val"),
            selection_metric=selection_metric,
        )
        if _is_improvement(selection_metric, best_metric, config.early_stopping_min_delta):
            best_metric = selection_metric
            epochs_without_improvement = 0
            best_checkpoint_path = str(
                save_checkpoint(
                    trainer,
                    checkpoint_dir / "best.pt",
                    epoch=epoch + 1,
                    global_step=global_step,
                    batch_index=None,
                    metrics=selection_metrics,
                    best_metric=best_metric,
                    metadata={"selection_metric_name": selection_metric_name},
                )
            )
        else:
            epochs_without_improvement += 1
        history.append(epoch_result)
        epoch_result["selection_metric"] = selection_metric
        epoch_result["epochs_without_improvement"] = epochs_without_improvement

        if (
            patience is not None
            and epochs_without_improvement >= patience
        ):
            stopped_early = True
            stopped_epoch = epoch + 1
            break

    if best_checkpoint_path is not None and "test" in dataloaders:
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        trainer.model.load_state_dict(state_dict)

    results: dict[str, object] = {
        "history": history,
        "available_splits": sorted(dataloaders.keys()),
        "checkpoint_dir": str(checkpoint_dir),
        "best_checkpoint": best_checkpoint_path,
        "train_batch_metrics_path": str(train_metrics_path),
        "epoch_metrics_path": str(epoch_metrics_path),
        "selection_metric_name": selection_metric_name,
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
    }
    if "test" in dataloaders:
        results["test"] = trainer.evaluate_epoch(dataloaders["test"])
    return results
