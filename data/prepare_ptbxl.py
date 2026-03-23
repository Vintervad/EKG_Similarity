from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PTB-XL 500 Hz records into the repository train/val/test format."
    )
    parser.add_argument(
        "--ptbxl-root",
        type=str,
        required=True,
        help="Path to the PTB-XL root directory containing ptbxl_database.csv and records500/.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data",
        help="Output dataset root in repo format. Defaults to 'data'.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional maximum number of train records to export.",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Optional maximum number of validation records to export.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional maximum number of test records to export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used if per-split sampling limits are set.",
    )
    return parser.parse_args()


def _resolve_repo_relative_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_ptbxl_metadata(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"PTB-XL metadata CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"PTB-XL metadata CSV {csv_path} has no header row.")
        required_columns = {"ecg_id", "filename_hr", "strat_fold"}
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise KeyError(
                f"PTB-XL metadata CSV {csv_path} is missing required columns: {sorted(missing)}"
            )
        return [row for row in reader]


def _split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for row in rows:
        fold = int(row["strat_fold"])
        if 1 <= fold <= 8:
            split_rows["train"].append(row)
        elif fold == 9:
            split_rows["val"].append(row)
        elif fold == 10:
            split_rows["test"].append(row)
        else:
            raise ValueError(f"Unexpected strat_fold value {fold!r} for ecg_id={row.get('ecg_id')}.")
    return split_rows


def _maybe_sample_rows(
    rows: list[dict[str, str]],
    max_records: int | None,
    rng: random.Random,
) -> list[dict[str, str]]:
    if max_records is None or max_records >= len(rows):
        return sorted(rows, key=lambda row: int(row["ecg_id"]))
    sampled = rng.sample(rows, k=max_records)
    return sorted(sampled, key=lambda row: int(row["ecg_id"]))


def _load_ptbxl_record(record_path: Path) -> np.ndarray:
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "The PTB-XL conversion script requires the 'wfdb' package. "
            "Install it with `pip install wfdb` in your environment."
        ) from exc

    base_record_path = record_path.with_suffix("") if record_path.suffix else record_path
    signal, _ = wfdb.rdsamp(str(base_record_path))
    if signal.ndim != 2:
        raise ValueError(f"Expected a 2D PTB-XL signal at {record_path}, got shape {signal.shape}.")
    return np.asarray(signal, dtype=np.float32)


def _prepare_output_root(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "raw" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)


def _write_split(
    split: str,
    rows: list[dict[str, str]],
    *,
    ptbxl_root: Path,
    output_root: Path,
) -> int:
    metadata_path = output_root / "metadata" / f"{split}.csv"
    raw_split_dir = output_root / "raw" / split
    written = 0

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["id", "path", "patient_id", "strat_fold", "filename_hr"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            ecg_id = int(row["ecg_id"])
            filename_hr = row["filename_hr"]
            record_path = ptbxl_root / filename_hr
            signal = _load_ptbxl_record(record_path)
            signal = signal.T

            output_filename = f"ecg_{ecg_id:05d}.npy"
            output_path = raw_split_dir / output_filename
            np.save(output_path, signal, allow_pickle=False)

            writer.writerow(
                {
                    "id": str(ecg_id),
                    "path": f"raw/{split}/{output_filename}",
                    "patient_id": row.get("patient_id", ""),
                    "strat_fold": row["strat_fold"],
                    "filename_hr": filename_hr,
                }
            )
            written += 1

    return written


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    ptbxl_root = Path(args.ptbxl_root).expanduser().resolve()
    output_root = _resolve_repo_relative_path(args.output_root)

    metadata_rows = _read_ptbxl_metadata(ptbxl_root / "ptbxl_database.csv")
    split_rows = _split_rows(metadata_rows)

    selected_rows = {
        "train": _maybe_sample_rows(split_rows["train"], args.max_train, rng),
        "val": _maybe_sample_rows(split_rows["val"], args.max_val, rng),
        "test": _maybe_sample_rows(split_rows["test"], args.max_test, rng),
    }

    _prepare_output_root(output_root)

    print(f"ptbxl_root={ptbxl_root}")
    print(f"output_root={output_root}")
    for split in ("train", "val", "test"):
        print(f"{split}_selected={len(selected_rows[split])}")

    for split in ("train", "val", "test"):
        count = _write_split(
            split,
            selected_rows[split],
            ptbxl_root=ptbxl_root,
            output_root=output_root,
        )
        print(f"{split}_written={count}")

    print("done=true")


if __name__ == "__main__":
    main()
