from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import ECGDataConfig, build_split_dataset, resolve_data_root
from models.encoder import ECGEncoderConfig
from utils.retrieval import (
    RetrievalIndex,
    build_model_for_retrieval,
    build_retrieval_index,
    save_retrieval_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed ECG dataset splits with the best checkpoint and save retrieval indices."
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["all"],
        choices=["all", "train", "val", "test"],
        help="Which splits to embed. Use 'all' to embed every available split.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embedding-type", type=str, default="global", choices=["retrieval", "global", "local"])
    parser.add_argument("--output-dir", type=str, default="embeddings")
    return parser.parse_args()


def _resolve_checkpoint_path(requested_checkpoint: str | None) -> str:
    if requested_checkpoint is not None:
        return requested_checkpoint
    default_checkpoint = Path("checkpoints") / "best.pt"
    if default_checkpoint.exists():
        return str(default_checkpoint)
    raise FileNotFoundError(
        "No checkpoint was provided and checkpoints/best.pt does not exist. Train the model first or pass --checkpoint."
    )


def _resolve_splits(data_root: Path, requested_splits: list[str]) -> list[str]:
    split_to_csv = {
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv",
    }
    if "all" not in requested_splits:
        return requested_splits

    available_splits = [
        split for split, filename in split_to_csv.items() if (data_root / "metadata" / filename).exists()
    ]
    if not available_splits:
        raise FileNotFoundError(f"No split CSV files were found under {data_root / 'metadata'}.")
    return available_splits


def _combined_index_name(splits: list[str], embedding_type: str) -> str:
    if splits == ["train", "val", "test"] or splits == ["train", "test", "val"]:
        prefix = "all"
    else:
        prefix = "_".join(splits)
    return f"{prefix}_{embedding_type}_index.pt"


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    selected_splits = _resolve_splits(data_root, args.splits)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ECGDataConfig(
        data_root=data_root,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_leads=args.channels,
    )
    model = build_model_for_retrieval(
        checkpoint_path=checkpoint_path,
        device=args.device,
        model_config=ECGEncoderConfig(input_channels=args.channels),
    )

    combined_embeddings: list[torch.Tensor] = []
    combined_ids: list[str] = []
    combined_paths: list[str] = []
    combined_splits: list[str] = []

    for split in selected_splits:
        dataset = build_split_dataset(config, split)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        index = build_retrieval_index(
            model=model,
            dataloader=dataloader,
            split=split,
            device=args.device,
            embedding_type=args.embedding_type,
        )
        split_output = output_dir / f"{split}_{args.embedding_type}_index.pt"
        save_retrieval_index(index, split_output)
        print(
            f"saved_split_index={split_output}, samples={len(index.ids)}, split={split}, embedding_type={args.embedding_type}"
        )
        combined_embeddings.append(index.embeddings)
        combined_ids.extend(index.ids)
        combined_paths.extend(index.paths)
        combined_splits.extend(index.splits)

    combined_index = RetrievalIndex(
        embeddings=torch.cat(combined_embeddings, dim=0),
        ids=combined_ids,
        paths=combined_paths,
        splits=combined_splits,
        source_name="+".join(selected_splits),
        embedding_type=args.embedding_type,
    )
    combined_output = output_dir / _combined_index_name(selected_splits, args.embedding_type)
    save_retrieval_index(combined_index, combined_output)
    print(f"checkpoint={checkpoint_path}")
    print(f"saved_combined_index={combined_output}, samples={len(combined_index.ids)}")


if __name__ == "__main__":
    main()
