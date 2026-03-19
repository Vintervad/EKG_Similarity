from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from data.dataset import ECGDataConfig, build_split_dataset, resolve_data_root
from models.encoder import ECGEncoderConfig
from utils.retrieval import (
    build_model_for_retrieval,
    build_multi_split_retrieval_index,
    build_retrieval_index,
    extract_embeddings,
    load_retrieval_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve similar ECGs with kNN in the learned embedding space.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--reference-split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--query-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--reference-index", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-queries", type=int, default=5)
    parser.add_argument("--embedding-type", type=str, default="global", choices=["retrieval", "global", "local"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_checkpoint = Path("checkpoints") / "best.pt"
        if default_checkpoint.exists():
            checkpoint_path = str(default_checkpoint)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint was provided and checkpoints/best.pt does not exist. Train the model first or pass --checkpoint."
        )

    config = ECGDataConfig(
        data_root=data_root,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_leads=args.channels,
    )

    if args.reference_index is not None:
        index = load_retrieval_index(args.reference_index)
        model = build_model_for_retrieval(
            checkpoint_path=checkpoint_path,
            device=args.device,
            model_config=ECGEncoderConfig(input_channels=args.channels),
        )
    else:
        model = build_model_for_retrieval(
            checkpoint_path=checkpoint_path,
            device=args.device,
            model_config=ECGEncoderConfig(input_channels=args.channels),
        )
        if args.reference_split == "all":
            index = build_multi_split_retrieval_index(
                data_root=data_root,
                splits=None,
                checkpoint_path=checkpoint_path,
                device=args.device,
                batch_size=args.batch_size,
                num_leads=args.channels,
                embedding_type=args.embedding_type,
            )
        else:
            reference_dataset = build_split_dataset(config, args.reference_split)
            reference_loader = DataLoader(reference_dataset, batch_size=args.batch_size, shuffle=False)

            index = build_retrieval_index(
                model=model,
                dataloader=reference_loader,
                split=args.reference_split,
                device=args.device,
                embedding_type=args.embedding_type,
            )

    query_dataset = build_split_dataset(config, args.query_split)
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False)
    query_payload = extract_embeddings(model, query_loader, device=args.device, embedding_type=args.embedding_type)
    query_results = index.query(query_payload["embeddings"], top_k=args.top_k)

    limit = min(args.limit_queries, len(query_results))
    print(f"checkpoint={checkpoint_path}")
    if args.reference_index is not None:
        print(f"reference_index={args.reference_index}")
    print(f"reference_source={index.source_name}")
    for query_idx in range(limit):
        query_id = query_payload["ids"][query_idx]
        query_path = query_payload["paths"][query_idx]
        print(f"query_index={query_idx}, query_id={query_id}, query_path={query_path}")
        for rank, result in enumerate(query_results[query_idx], start=1):
            print(
                f"  rank={rank}, score={result['score']:.4f}, "
                f"id={result['id']}, split={result['split']}, path={result['path']}"
            )


if __name__ == "__main__":
    main()
