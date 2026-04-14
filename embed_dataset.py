import argparse
from pathlib import Path

import torch
from data.dataset import ECGDataConfig, build_split_dataset, resolve_data_root
from models.encoder import ECGEncoderConfig
from torch.utils.data import DataLoader
from utils.faiss_engine import FAISSGPUEngine
from utils.retrieval import build_model_for_retrieval, extract_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed the ECG dataset into a FAISS GPU index."
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["all"],
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="embeddings")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--embedding-type", type=str, default="global")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    output_dir = Path(args.output_dir)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_checkpoint = Path("checkpoints") / "best.pt"
        if default_checkpoint.exists():
            checkpoint_path = str(default_checkpoint)
        else:
            raise FileNotFoundError("Train the model first or pass --checkpoint.")

    model = build_model_for_retrieval(
        checkpoint_path=checkpoint_path,
        device=args.device,
        model_config=ECGEncoderConfig(input_channels=args.channels),
    )

    config = ECGDataConfig(
        data_root=data_root,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_leads=args.channels,
    )

    splits_to_process = (
        ["train", "val", "test"] if "all" in args.splits else args.splits
    )

    # Initialize the FAISS Engine. Dimension is usually extracted from the first batch,
    # but we can initialize it dynamically after extracting the first split.
    combined_engine = None
    use_gpu = args.device == "cuda"

    for split in splits_to_process:
        print(f"Extracting embeddings for {split} split...")
        dataset = build_split_dataset(config, split)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        payload = extract_embeddings(
            model, loader, device=args.device, embedding_type=args.embedding_type
        )

        dim = payload["embeddings"].shape[1]
        split_engine = FAISSGPUEngine(dimension=dim, use_gpu=use_gpu)
        split_splits = [split] * len(payload["ids"])

        split_engine.add_embeddings(
            payload["embeddings"], payload["ids"], payload["paths"], split_splits
        )
        split_engine.save(output_dir, f"{split}_{args.embedding_type}")

        if combined_engine is None:
            combined_engine = FAISSGPUEngine(dimension=dim, use_gpu=use_gpu)

        combined_engine.add_embeddings(
            payload["embeddings"], payload["ids"], payload["paths"], split_splits
        )

    if "all" in args.splits or len(splits_to_process) > 1:
        print(f"Saving combined FAISS index...")
        combined_engine.save(output_dir, f"all_{args.embedding_type}")


if __name__ == "__main__":
    main()
