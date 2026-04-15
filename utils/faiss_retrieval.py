from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _require_faiss() -> Any:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for the FAISS retrieval backend. Install a compatible FAISS build "
            "for your environment before using --backend faiss."
        ) from exc
    return faiss


def _to_numpy_float32(embeddings: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(embeddings, torch.Tensor):
        array = embeddings.detach().cpu().numpy()
    else:
        array = np.asarray(embeddings)
    return np.ascontiguousarray(array, dtype=np.float32)


def _metadata_path_for(index_path: str | Path) -> Path:
    index_path = Path(index_path)
    return index_path.with_name(f"{index_path.stem}_meta.pt")


def _move_index_to_cpu(index: Any, faiss: Any) -> Any:
    if hasattr(faiss, "index_gpu_to_cpu"):
        try:
            return faiss.index_gpu_to_cpu(index)
        except Exception:
            return index
    return index


def _maybe_move_index_to_gpu(index: Any, use_gpu: bool, gpu_device: int, faiss: Any) -> tuple[Any, Any | None]:
    if not use_gpu:
        return index, None
    if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
        raise RuntimeError(
            "The installed FAISS build does not expose GPU support. "
            "Install a GPU-enabled FAISS build or rerun without --faiss-use-gpu."
        )
    resources = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(resources, gpu_device, index)
    return gpu_index, resources


def _training_subset(
    embeddings: np.ndarray,
    train_size: int,
    train_seed: int,
) -> np.ndarray:
    if len(embeddings) == 0:
        raise ValueError("Cannot train a FAISS index with zero embeddings.")
    if train_size <= 0 or train_size >= len(embeddings):
        return embeddings
    rng = np.random.default_rng(train_seed)
    indices = rng.choice(len(embeddings), size=train_size, replace=False)
    return embeddings[indices]


def _build_cpu_index(
    dim: int,
    index_type: str,
    nlist: int,
    pq_m: int,
    pq_bits: int,
    faiss: Any,
) -> Any:
    metric = faiss.METRIC_INNER_PRODUCT
    if index_type == "flat":
        return faiss.IndexFlatIP(dim)

    if nlist < 1:
        raise ValueError("nlist must be >= 1 for IVF-based FAISS indices.")
    quantizer = faiss.IndexFlatIP(dim)
    if index_type == "ivf-flat":
        return faiss.IndexIVFFlat(quantizer, dim, nlist, metric)
    if index_type == "ivf-pq":
        if dim % pq_m != 0:
            raise ValueError(
                f"faiss_pq_m must divide the embedding dimension. Got dim={dim} and faiss_pq_m={pq_m}."
            )
        return faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_bits, metric)
    raise ValueError("index_type must be one of {'flat', 'ivf-flat', 'ivf-pq'}.")


def _set_nprobe(index: Any, nprobe: int | None) -> int | None:
    if nprobe is None or not hasattr(index, "nprobe"):
        return None
    index.nprobe = int(nprobe)
    return int(index.nprobe)


@dataclass
class FaissRetrievalIndex:
    index: Any
    ids: list[str]
    paths: list[str]
    splits: list[str]
    source_name: str
    embedding_type: str
    index_type: str
    normalize: bool = True
    nprobe: int | None = None
    gpu_resources: Any | None = field(default=None, repr=False)

    def set_nprobe(self, nprobe: int) -> None:
        updated_nprobe = _set_nprobe(self.index, nprobe)
        self.nprobe = updated_nprobe if updated_nprobe is not None else self.nprobe

    def query(self, query_embeddings: torch.Tensor | np.ndarray, top_k: int = 5) -> list[list[dict[str, Any]]]:
        queries = _to_numpy_float32(query_embeddings)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        top_k = min(top_k, len(self.ids))
        scores, indices = self.index.search(queries, top_k)

        results: list[list[dict[str, Any]]] = []
        for score_row, index_row in zip(scores, indices):
            row: list[dict[str, Any]] = []
            for score, idx in zip(score_row.tolist(), index_row.tolist()):
                if idx < 0:
                    continue
                row.append(
                    {
                        "index": idx,
                        "id": self.ids[idx],
                        "path": self.paths[idx],
                        "split": self.splits[idx],
                        "score": float(score),
                    }
                )
            results.append(row)
        return results


def build_faiss_retrieval_index(
    embeddings: torch.Tensor | np.ndarray,
    ids: list[str],
    paths: list[str],
    splits: list[str],
    source_name: str,
    embedding_type: str = "global",
    *,
    index_type: str = "ivf-flat",
    use_gpu: bool = False,
    gpu_device: int = 0,
    nlist: int = 65536,
    nprobe: int = 64,
    train_size: int = 200000,
    train_seed: int = 0,
    pq_m: int = 16,
    pq_bits: int = 8,
    add_batch_size: int = 100000,
) -> FaissRetrievalIndex:
    faiss = _require_faiss()
    embedding_array = _to_numpy_float32(embeddings)
    if embedding_array.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [num_samples, dim], got {embedding_array.shape}.")
    if len(ids) != len(paths) or len(ids) != len(splits) or len(ids) != embedding_array.shape[0]:
        raise ValueError("Embeddings, ids, paths, and splits must all have the same length.")
    if add_batch_size < 1:
        raise ValueError("add_batch_size must be >= 1.")

    dim = int(embedding_array.shape[1])
    effective_train_size = len(embedding_array) if train_size <= 0 else min(train_size, len(embedding_array))
    effective_nlist = min(nlist, max(1, effective_train_size))
    cpu_index = _build_cpu_index(
        dim,
        index_type=index_type,
        nlist=effective_nlist,
        pq_m=pq_m,
        pq_bits=pq_bits,
        faiss=faiss,
    )
    if hasattr(cpu_index, "is_trained") and not cpu_index.is_trained:
        train_embeddings = _training_subset(embedding_array, train_size=effective_train_size, train_seed=train_seed)
        cpu_index.train(train_embeddings)

    index, gpu_resources = _maybe_move_index_to_gpu(cpu_index, use_gpu=use_gpu, gpu_device=gpu_device, faiss=faiss)
    effective_nprobe = _set_nprobe(index, nprobe)

    for start in range(0, embedding_array.shape[0], add_batch_size):
        batch = embedding_array[start : start + add_batch_size]
        index.add(batch)

    return FaissRetrievalIndex(
        index=index,
        ids=list(ids),
        paths=list(paths),
        splits=list(splits),
        source_name=source_name,
        embedding_type=embedding_type,
        index_type=index_type,
        normalize=True,
        nprobe=effective_nprobe,
        gpu_resources=gpu_resources,
    )


def save_faiss_retrieval_index(index: FaissRetrievalIndex, output_path: str | Path) -> tuple[Path, Path]:
    faiss = _require_faiss()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_index = _move_index_to_cpu(index.index, faiss)
    faiss.write_index(cpu_index, str(output_path))

    metadata_path = _metadata_path_for(output_path)
    torch.save(
        {
            "ids": index.ids,
            "paths": index.paths,
            "splits": index.splits,
            "source_name": index.source_name,
            "embedding_type": index.embedding_type,
            "index_type": index.index_type,
            "normalize": index.normalize,
            "nprobe": index.nprobe,
        },
        metadata_path,
    )
    return output_path, metadata_path


def load_faiss_retrieval_index(
    index_path: str | Path,
    *,
    use_gpu: bool = False,
    gpu_device: int = 0,
    nprobe: int | None = None,
) -> FaissRetrievalIndex:
    faiss = _require_faiss()
    index_path = Path(index_path)
    metadata_path = _metadata_path_for(index_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"FAISS metadata file not found: {metadata_path}. Expected it next to the FAISS index."
        )

    metadata = torch.load(metadata_path, map_location="cpu")
    index = faiss.read_index(str(index_path))
    index, gpu_resources = _maybe_move_index_to_gpu(index, use_gpu=use_gpu, gpu_device=gpu_device, faiss=faiss)
    effective_nprobe = _set_nprobe(index, nprobe if nprobe is not None else metadata.get("nprobe"))

    return FaissRetrievalIndex(
        index=index,
        ids=list(metadata["ids"]),
        paths=list(metadata["paths"]),
        splits=list(metadata["splits"]),
        source_name=str(metadata.get("source_name", index_path.stem)),
        embedding_type=str(metadata.get("embedding_type", "global")),
        index_type=str(metadata.get("index_type", "ivf-flat")),
        normalize=bool(metadata.get("normalize", True)),
        nprobe=effective_nprobe,
        gpu_resources=gpu_resources,
    )
