from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from data.augmentations import TwoViewECGAugmentor
from losses.total_loss import ECGTrainingObjective
from models.encoder import ECGContrastiveAutoencoder


class ContrastiveAutoencoderTrainer:
    def __init__(
        self,
        model: ECGContrastiveAutoencoder,
        objective: ECGTrainingObjective,
        optimizer: torch.optim.Optimizer,
        augmentor: TwoViewECGAugmentor | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.augmentor = augmentor
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.objective.to(self.device)

    def _move_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(self.device, non_blocking=True)

    def _prepare_batch(
        self,
        batch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch, torch.Tensor):
            signal = self._move_tensor(batch)
            if self.augmentor is None:
                raise ValueError("An augmentor is required when the batch only contains raw ECG signals.")
            view1, view2 = self.augmentor(signal)
            return view1, view2, view1, view2

        if isinstance(batch, Mapping):
            if "view1" in batch and "view2" in batch:
                view1 = self._move_tensor(batch["view1"])
                view2 = self._move_tensor(batch["view2"])
                target1 = self._move_tensor(batch.get("target1", batch["view1"]))
                target2 = self._move_tensor(batch.get("target2", batch["view2"]))
                return view1, view2, target1, target2
            if "signal" in batch:
                signal = self._move_tensor(batch["signal"])
                if self.augmentor is None:
                    raise ValueError("An augmentor is required when the batch provides only `signal`.")
                view1, view2 = self.augmentor(signal)
                return view1, view2, view1, view2
            raise KeyError("Batch mapping must contain either `signal` or both `view1` and `view2`.")

        if isinstance(batch, Sequence) and len(batch) == 2 and all(isinstance(item, torch.Tensor) for item in batch):
            view1 = self._move_tensor(batch[0])
            view2 = self._move_tensor(batch[1])
            return view1, view2, view1, view2

        raise TypeError("Unsupported batch format. Use a tensor, a mapping, or a two-tensor sequence.")

    def step(
        self,
        batch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        train: bool = True,
    ) -> dict[str, float]:
        view1, view2, target1, target2 = self._prepare_batch(batch)
        self.model.train(train)
        with torch.set_grad_enabled(train):
            outputs_view1 = self.model(view1)
            outputs_view2 = self.model(view2)
            losses = self.objective(outputs_view1, outputs_view2, target1, target2)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                self.optimizer.step()

        metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
        metrics["batch_size"] = float(view1.size(0))
        return metrics

    def fit_epoch(self, data_loader: Any) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for batch in data_loader:
            metrics = self.step(batch, train=True)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        if count == 0:
            raise ValueError("data_loader is empty.")
        return {key: value / count for key, value in totals.items()}

    def evaluate_epoch(self, data_loader: Any) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for batch in data_loader:
            metrics = self.step(batch, train=False)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        if count == 0:
            raise ValueError("data_loader is empty.")
        return {key: value / count for key, value in totals.items()}
