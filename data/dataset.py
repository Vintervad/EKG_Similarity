from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
ECG_LEAD_COLUMNS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
ID_COLUMN_CANDIDATES = ("id", "Test_ID", "test_id", "testId", "TestID")


@dataclass
class ECGDataConfig:
    data_root: str | Path = "data"
    train_csv: str | Path = "train.csv"
    val_csv: str | Path = "val.csv"
    test_csv: str | Path = "test.csv"
    signal_column: str = "path"
    id_column: str | None = "id"
    num_leads: int = 12
    batch_size: int = 8
    eval_batch_size: int | None = None
    num_workers: int = 0
    pin_memory: bool = False
    drop_last_train: bool = False
    target_length: int | None = None


def resolve_data_root(data_root: str | Path) -> Path:
    path = Path(data_root)
    return path if path.is_absolute() else REPO_ROOT / path


class ECGDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        data_root: str | Path | None = None,
        signal_column: str = "path",
        id_column: str | None = "id",
        num_leads: int = 12,
        target_length: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.split = split
        self.signal_column = signal_column
        self.id_column = id_column
        self.num_leads = num_leads
        self.target_length = target_length
        self.data_root = (
            resolve_data_root(data_root)
            if data_root is not None
            else self.csv_path.parent.parent
        )
        self.raw_split_dir = self.data_root / "raw" / split
        self.records = self._read_records()

    def _read_records(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {self.csv_path} has no header row.")
            if self.signal_column not in reader.fieldnames:
                raise KeyError(
                    f"CSV file {self.csv_path} must contain the column {self.signal_column!r}. "
                    f"Found columns: {reader.fieldnames}"
                )
            records = [row for row in reader if row.get(self.signal_column)]
        if not records:
            raise ValueError(
                f"CSV file {self.csv_path} does not contain any rows with {self.signal_column!r}."
            )
        return records

    def _resolve_path(self, value: str) -> Path:
        raw_path = Path(value)
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend(
                [
                    raw_path,
                    self.data_root / raw_path,
                    REPO_ROOT / raw_path,
                    self.csv_path.parent / raw_path,
                    self.raw_split_dir / raw_path,
                ]
            )
        seen: set[Path] = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not resolve ECG file path {value!r} from CSV {self.csv_path}. Checked: {unique_candidates}"
        )

    def _load_array(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path, allow_pickle=False)
        elif suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict):
                if "signal" not in payload:
                    raise KeyError(
                        f"Tensor file {path} is a dict but does not contain a 'signal' key."
                    )
                payload = payload["signal"]
            if not isinstance(payload, torch.Tensor):
                raise TypeError(
                    f"Expected torch.Tensor inside {path}, got {type(payload)!r}."
                )
            array = payload.detach().cpu().numpy()
        elif suffix == ".parquet":
            array = self._load_parquet_array(path)
        else:
            raise ValueError(
                f"Unsupported ECG file type {suffix!r}. Use .npy, .pt, .pth, or .parquet."
            )
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Loaded object from {path} is not a NumPy array.")
        return array

    def _load_parquet_array(self, path: Path) -> np.ndarray:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Reading parquet ECG files requires pandas with a parquet engine "
                "such as pyarrow or fastparquet."
            ) from exc

        frame = pd.read_parquet(path)
        missing_columns = [
            column for column in ECG_LEAD_COLUMNS if column not in frame.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Parquet file {path} is missing ECG lead columns {missing_columns}. "
                f"Found columns: {list(frame.columns)}"
            )

        array = frame.loc[:, ECG_LEAD_COLUMNS].to_numpy(dtype=np.float32).T
        return array / 1000.0

    def _to_leads_time_tensor(self, array: np.ndarray, path: Path) -> torch.Tensor:
        if array.ndim < 2:
            raise ValueError(
                f"ECG array from {path} must have at least 2 dimensions, got shape {array.shape}."
            )

        lead_axes = [
            index for index, size in enumerate(array.shape) if size == self.num_leads
        ]
        if len(lead_axes) != 1:
            raise ValueError(
                f"ECG array from {path} must have exactly one axis of size {self.num_leads}. "
                f"Got shape {array.shape}."
            )

        lead_axis = lead_axes[0]
        array = np.moveaxis(array, lead_axis, 0)
        array = array.reshape(self.num_leads, -1)
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        return tensor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        signal_path = self._resolve_path(record[self.signal_column])
        array = self._load_array(signal_path)
        signal = self._to_leads_time_tensor(array, signal_path)

        if self.target_length is not None:
            curr_len = signal.size(-1)
            if curr_len < self.target_length:
                padding = torch.zeros(signal.size(0), self.target_length - curr_len, dtype=signal.dtype)
                signal = torch.cat([signal, padding], dim=-1)
            elif curr_len > self.target_length:
                signal = signal[..., : self.target_length]

        sample: dict[str, Any] = {
            "signal": signal,
            "path": str(signal_path),
        }
        if self.id_column is not None and self.id_column in record:
            sample["id"] = record[self.id_column]
        elif self.id_column is not None:
            for candidate in ID_COLUMN_CANDIDATES:
                if candidate in record:
                    sample["id"] = record[candidate]
                    break
        return sample


def _build_split_csv_path(data_root: Path, filename: str | Path) -> Path:
    csv_path = Path(filename)
    if csv_path.is_absolute():
        return csv_path
    direct_candidate = REPO_ROOT / csv_path
    if csv_path.exists():
        return csv_path.resolve()
    if direct_candidate.exists():
        return direct_candidate
    return data_root / "metadata" / csv_path


def build_split_dataset(config: ECGDataConfig, split: str) -> ECGDataset:
    filename_map = {
        "train": config.train_csv,
        "val": config.val_csv,
        "test": config.test_csv,
    }
    if split not in filename_map:
        raise ValueError(
            f"Unsupported split {split!r}. Expected one of {sorted(filename_map)}."
        )
    data_root = resolve_data_root(config.data_root)
    csv_path = _build_split_csv_path(data_root, filename_map[split])
    return ECGDataset(
        csv_path=csv_path,
        split=split,
        data_root=data_root,
        signal_column=config.signal_column,
        id_column=config.id_column,
        num_leads=config.num_leads,
        target_length=config.target_length,
    )


def build_split_dataloaders(config: ECGDataConfig) -> dict[str, DataLoader]:
    data_root = resolve_data_root(config.data_root)
    eval_batch_size = config.eval_batch_size or config.batch_size
    settings = {
        "train": {
            "batch_size": config.batch_size,
            "shuffle": True,
            "drop_last": config.drop_last_train,
        },
        "val": {
            "batch_size": eval_batch_size,
            "shuffle": False,
            "drop_last": False,
        },
        "test": {
            "batch_size": eval_batch_size,
            "shuffle": False,
            "drop_last": False,
        },
    }

    dataloaders: dict[str, DataLoader] = {}
    for split, loader_kwargs in settings.items():
        csv_name = {
            "train": config.train_csv,
            "val": config.val_csv,
            "test": config.test_csv,
        }[split]
        csv_path = _build_split_csv_path(data_root, csv_name)
        if not csv_path.exists():
            if split == "train":
                raise FileNotFoundError(f"Training split CSV not found: {csv_path}")
            continue

        dataset = build_split_dataset(config, split)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=loader_kwargs["batch_size"],
            shuffle=loader_kwargs["shuffle"],
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=loader_kwargs["drop_last"],
        )
    return dataloaders
