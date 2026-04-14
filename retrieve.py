from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset import ECGDataConfig, build_split_dataset, resolve_data_root
from models.encoder import ECGEncoderConfig
from torch.utils.data import DataLoader
from utils.faiss_engine import FAISSEngine
from utils.retrieval import build_model_for_retrieval, extract_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve similar ECGs using FAISS GPU."
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--reference-split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument(
        "--query-split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--index-dir", type=str, default="embeddings")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=5)  # Restored original default
    parser.add_argument("--limit-queries", type=int, default=5)
    parser.add_argument("--embedding-type", type=str, default="global")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    index_dir = Path(args.index_dir)
    use_gpu = args.device == "cuda"

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_checkpoint = Path("checkpoints") / "best.pt"
        if default_checkpoint.exists():
            checkpoint_path = str(default_checkpoint)
        else:
            raise FileNotFoundError("Train the model first or pass --checkpoint.")

    config = ECGDataConfig(
        data_root=data_root,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_leads=args.channels,
    )

    # 1. Load the FAISS Engine
    prefix = f"{args.reference_split}_{args.embedding_type}"
    print(f"Loading FAISS reference index: {prefix} from {index_dir}")
    faiss_engine = FAISSEngine.load(index_dir, prefix, use_gpu=use_gpu)

    # 2. Build model and extract query embeddings
    model = build_model_for_retrieval(
        checkpoint_path=checkpoint_path,
        device=args.device,
        model_config=ECGEncoderConfig(input_channels=args.channels),
    )

    query_dataset = build_split_dataset(config, args.query_split)
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False)
    query_payload = extract_embeddings(
        model, query_loader, device=args.device, embedding_type=args.embedding_type
    )

    # 3. Execute FAISS Retrieval
    print(f"Querying top {args.top_k} nearest neighbors via FAISS...")
    query_results = faiss_engine.search(query_payload["embeddings"], top_k=args.top_k)

    # 4. Process Output (Restored original exact loop)
    limit = min(args.limit_queries, len(query_results))
    print(f"checkpoint={checkpoint_path}")
    print(f"reference_source=FAISS_{prefix}")

    for query_idx in range(limit):
        query_id = query_payload["ids"][query_idx]
        query_path = query_payload["paths"][query_idx]
        print(
            f"\nquery_index={query_idx}, query_id={query_id}, query_path={query_path}"
        )

        for rank, result in enumerate(query_results[query_idx], start=1):
            print(
                f"  rank={rank}, score={result['score']:.4f}, "
                f"id={result['id']}, split={result['split']}, path={result['path']}"
            )


if __name__ == "__main__":
    main()
