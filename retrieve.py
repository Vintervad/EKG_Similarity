from __future__ import annotations

import argparse
import csv
from pathlib import Path

from torch.utils.data import DataLoader

from data.dataset import ECGDataConfig, build_split_dataset, resolve_data_root
from models.encoder import ECGEncoderConfig
from utils.faiss_retrieval import build_faiss_retrieval_index, load_faiss_retrieval_index
from utils.retrieval import (
    build_model_for_retrieval,
    build_multi_split_retrieval_index,
    build_retrieval_index,
    extract_embeddings,
    load_retrieval_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve similar ECGs with kNN or FAISS ANN in the learned embedding space.")
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
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="global",
        choices=["retrieval", "global", "projection", "local"],
    )
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "faiss"])
    parser.add_argument(
        "--faiss-index-type",
        type=str,
        default="ivf-flat",
        choices=["flat", "ivf-flat", "ivf-pq"],
        help="FAISS index family to build when --backend faiss is used and no saved reference index is provided.",
    )
    parser.add_argument(
        "--faiss-use-gpu",
        action="store_true",
        help="Use a GPU-backed FAISS index when a GPU-enabled FAISS build is installed.",
    )
    parser.add_argument("--faiss-gpu-device", type=int, default=0)
    parser.add_argument("--faiss-nlist", type=int, default=65536)
    parser.add_argument("--faiss-nprobe", type=int, default=64)
    parser.add_argument("--faiss-train-size", type=int, default=200000)
    parser.add_argument("--faiss-train-seed", type=int, default=0)
    parser.add_argument("--faiss-pq-m", type=int, default=16)
    parser.add_argument("--faiss-pq-bits", type=int, default=8)
    parser.add_argument("--faiss-add-batch-size", type=int, default=100000)
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save all printed top-k retrieval results as a CSV file.",
    )
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
        model = build_model_for_retrieval(
            checkpoint_path=checkpoint_path,
            device=args.device,
            model_config=ECGEncoderConfig(input_channels=args.channels),
        )
        if args.backend == "faiss":
            index = load_faiss_retrieval_index(
                args.reference_index,
                use_gpu=args.faiss_use_gpu,
                gpu_device=args.faiss_gpu_device,
                nprobe=args.faiss_nprobe,
            )
        else:
            index = load_retrieval_index(args.reference_index)
    else:
        model = build_model_for_retrieval(
            checkpoint_path=checkpoint_path,
            device=args.device,
            model_config=ECGEncoderConfig(input_channels=args.channels),
        )
        if args.reference_split == "all":
            reference_index = build_multi_split_retrieval_index(
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

            reference_index = build_retrieval_index(
                model=model,
                dataloader=reference_loader,
                split=args.reference_split,
                device=args.device,
                embedding_type=args.embedding_type,
            )
        if args.backend == "faiss":
            index = build_faiss_retrieval_index(
                embeddings=reference_index.embeddings,
                ids=reference_index.ids,
                paths=reference_index.paths,
                splits=reference_index.splits,
                source_name=reference_index.source_name,
                embedding_type=reference_index.embedding_type,
                index_type=args.faiss_index_type,
                use_gpu=args.faiss_use_gpu,
                gpu_device=args.faiss_gpu_device,
                nlist=args.faiss_nlist,
                nprobe=args.faiss_nprobe,
                train_size=args.faiss_train_size,
                train_seed=args.faiss_train_seed,
                pq_m=args.faiss_pq_m,
                pq_bits=args.faiss_pq_bits,
                add_batch_size=args.faiss_add_batch_size,
            )
        else:
            index = reference_index

    query_dataset = build_split_dataset(config, args.query_split)
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False)
    query_payload = extract_embeddings(model, query_loader, device=args.device, embedding_type=args.embedding_type)
    query_results = index.query(query_payload["embeddings"], top_k=args.top_k)

    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "query_index",
                    "query_id",
                    "query_path",
                    "rank",
                    "neighbor_index",
                    "neighbor_id",
                    "neighbor_split",
                    "neighbor_path",
                    "score",
                ],
            )
            writer.writeheader()
            for query_idx, result_row in enumerate(query_results):
                query_id = query_payload["ids"][query_idx]
                query_path = query_payload["paths"][query_idx]
                for rank, result in enumerate(result_row, start=1):
                    writer.writerow(
                        {
                            "query_index": query_idx,
                            "query_id": query_id,
                            "query_path": query_path,
                            "rank": rank,
                            "neighbor_index": result["index"],
                            "neighbor_id": result["id"],
                            "neighbor_split": result["split"],
                            "neighbor_path": result["path"],
                            "score": result["score"],
                        }
                    )

    limit = min(args.limit_queries, len(query_results))
    print(f"checkpoint={checkpoint_path}")
    if args.reference_index is not None:
        print(f"reference_index={args.reference_index}")
    print(f"backend={args.backend}")
    print(f"reference_source={index.source_name}")
    if args.backend == "faiss":
        print(f"faiss_index_type={index.index_type}, nprobe={index.nprobe}")
    for query_idx in range(limit):
        query_id = query_payload["ids"][query_idx]
        query_path = query_payload["paths"][query_idx]
        print(f"query_index={query_idx}, query_id={query_id}, query_path={query_path}")
        for rank, result in enumerate(query_results[query_idx], start=1):
            print(
                f"  rank={rank}, score={result['score']:.4f}, "
                f"id={result['id']}, split={result['split']}, path={result['path']}"
            )
    if args.output_csv is not None:
        print(f"saved_topk_csv={args.output_csv}")


if __name__ == "__main__":
    main()
