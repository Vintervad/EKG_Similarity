from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import ECGDataConfig, build_split_dataset, resolve_data_root
from models.encoder import ECGContrastiveAutoencoder, ECGEncoderConfig


@dataclass
class RetrievalIndex:
    embeddings: torch.Tensor
    ids: list[str]
    paths: list[str]
    splits: list[str]
    source_name: str
    embedding_type: str = "global"

    def query(self, query_embeddings: torch.Tensor, top_k: int = 5) -> list[list[dict[str, Any]]]:
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.unsqueeze(0)
        query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=-1)
        reference_embeddings = torch.nn.functional.normalize(self.embeddings, dim=-1)
        similarities = query_embeddings @ reference_embeddings.T
        top_k = min(top_k, reference_embeddings.size(0))
        scores, indices = torch.topk(similarities, k=top_k, dim=1)

        results: list[list[dict[str, Any]]] = []
        for index_row, score_row in zip(indices, scores):
            query_results = []
            for index, score in zip(index_row.tolist(), score_row.tolist()):
                query_results.append(
                    {
                        "index": index,
                        "id": self.ids[index],
                        "path": self.paths[index],
                        "score": float(score),
                        "split": self.splits[index],
                    }
                )
            results.append(query_results)
        return results


def _extract_signals(batch: torch.Tensor | dict[str, Any]) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict) and "signal" in batch:
        return batch["signal"]
    raise TypeError("Expected a tensor batch or a dict containing the key 'signal'.")


def _extract_metadata(batch: torch.Tensor | dict[str, Any], batch_size: int) -> tuple[list[str], list[str]]:
    if not isinstance(batch, dict):
        ids = [f"sample_{index}" for index in range(batch_size)]
        return ids, [""] * batch_size

    ids = batch.get("id")
    paths = batch.get("path")
    if ids is None:
        ids = [f"sample_{index}" for index in range(batch_size)]
    elif isinstance(ids, str):
        ids = [ids]
    else:
        ids = [str(item) for item in ids]

    if paths is None:
        paths = [""] * batch_size
    elif isinstance(paths, str):
        paths = [paths]
    else:
        paths = [str(item) for item in paths]

    return ids, paths


def extract_embeddings(
    model: ECGContrastiveAutoencoder,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
    embedding_type: str = "global",
) -> dict[str, Any]:
    model = model.to(device)
    model.eval()

    all_embeddings: list[torch.Tensor] = []
    all_ids: list[str] = []
    all_paths: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            signals = _extract_signals(batch).to(device)
            embeddings = model.embed(signals, embedding_type=embedding_type, normalize=True)
            all_embeddings.append(embeddings.detach().cpu())
            ids, paths = _extract_metadata(batch, embeddings.size(0))
            all_ids.extend(ids)
            all_paths.extend(paths)

    if not all_embeddings:
        raise ValueError("Dataloader did not yield any batches.")

    return {
        "embeddings": torch.cat(all_embeddings, dim=0),
        "ids": all_ids,
        "paths": all_paths,
    }


def build_retrieval_index(
    model: ECGContrastiveAutoencoder,
    dataloader: DataLoader,
    split: str,
    device: torch.device | str = "cpu",
    embedding_type: str = "global",
) -> RetrievalIndex:
    payload = extract_embeddings(model, dataloader, device=device, embedding_type=embedding_type)
    return RetrievalIndex(
        embeddings=payload["embeddings"],
        ids=payload["ids"],
        paths=payload["paths"],
        splits=[split] * len(payload["ids"]),
        source_name=split,
        embedding_type=embedding_type,
    )


def build_model_for_retrieval(
    checkpoint_path: str | Path | None = None,
    device: torch.device | str = "cpu",
    model_config: ECGEncoderConfig | None = None,
) -> ECGContrastiveAutoencoder:
    model = ECGContrastiveAutoencoder(model_config or ECGEncoderConfig())
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError("Checkpoint must be a state dict or a dict containing 'model_state_dict'.")
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def build_split_retrieval_index(
    data_root: str | Path,
    split: str,
    checkpoint_path: str | Path | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 8,
    num_leads: int = 12,
    embedding_type: str = "global",
) -> RetrievalIndex:
    config = ECGDataConfig(
        data_root=resolve_data_root(data_root),
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_leads=num_leads,
    )
    dataset = build_split_dataset(config, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = build_model_for_retrieval(
        checkpoint_path=checkpoint_path,
        device=device,
        model_config=ECGEncoderConfig(input_channels=num_leads),
    )
    return build_retrieval_index(model, dataloader, split=split, device=device, embedding_type=embedding_type)


def _available_splits(config: ECGDataConfig) -> list[str]:
    data_root = resolve_data_root(config.data_root)
    split_to_csv = {
        "train": config.train_csv,
        "val": config.val_csv,
        "test": config.test_csv,
    }
    return [split for split, filename in split_to_csv.items() if (data_root / "metadata" / filename).exists()]


def build_multi_split_retrieval_index(
    data_root: str | Path,
    splits: list[str] | tuple[str, ...] | None = None,
    checkpoint_path: str | Path | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 8,
    num_leads: int = 12,
    embedding_type: str = "global",
) -> RetrievalIndex:
    config = ECGDataConfig(
        data_root=resolve_data_root(data_root),
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_leads=num_leads,
    )
    selected_splits = list(splits) if splits is not None else _available_splits(config)
    if not selected_splits:
        raise ValueError("No dataset splits were found to build the retrieval index.")

    model = build_model_for_retrieval(
        checkpoint_path=checkpoint_path,
        device=device,
        model_config=ECGEncoderConfig(input_channels=num_leads),
    )

    all_embeddings: list[torch.Tensor] = []
    all_ids: list[str] = []
    all_paths: list[str] = []
    all_splits: list[str] = []

    for split in selected_splits:
        dataset = build_split_dataset(config, split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        payload = extract_embeddings(model, dataloader, device=device, embedding_type=embedding_type)
        all_embeddings.append(payload["embeddings"])
        all_ids.extend(payload["ids"])
        all_paths.extend(payload["paths"])
        all_splits.extend([split] * len(payload["ids"]))

    return RetrievalIndex(
        embeddings=torch.cat(all_embeddings, dim=0),
        ids=all_ids,
        paths=all_paths,
        splits=all_splits,
        source_name="+".join(selected_splits),
        embedding_type=embedding_type,
    )


def save_retrieval_index(index: RetrievalIndex, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": index.embeddings,
            "ids": index.ids,
            "paths": index.paths,
            "splits": index.splits,
            "source_name": index.source_name,
            "embedding_type": index.embedding_type,
        },
        output_path,
    )
    return output_path


def load_retrieval_index(index_path: str | Path) -> RetrievalIndex:
    payload = torch.load(index_path, map_location="cpu")
    required_keys = {"embeddings", "ids", "paths", "splits"}
    if not isinstance(payload, dict) or not required_keys.issubset(payload):
        raise ValueError(
            f"Retrieval index at {index_path} must contain the keys {sorted(required_keys)}."
        )
    return RetrievalIndex(
        embeddings=payload["embeddings"],
        ids=list(payload["ids"]),
        paths=list(payload["paths"]),
        splits=list(payload["splits"]),
        source_name=str(payload.get("source_name", "loaded_index")),
        embedding_type=str(payload.get("embedding_type", "global")),
    )
