from pathlib import Path

import faiss
import numpy as np
import torch


class FAISSEngine:
    def __init__(self, dimension: int, use_gpu: bool = True):
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.metadata = {"ids": [], "paths": [], "splits": []}

        # IndexFlatIP calculates inner product (Cosine Similarity for normalized vectors)
        self.cpu_index = faiss.IndexFlatIP(dimension)

        if self.use_gpu:
            self.res = faiss.StandardGpuResources()
            # Move index to GPU
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        else:
            self.index = self.cpu_index

    def add_batch(self, embeddings: torch.Tensor, ids: list, paths: list, splits: list):
        """Converts tensors to float32 numpy arrays and adds them to the FAISS index."""
        emb_np = embeddings.cpu().numpy().astype(np.float32)
        self.index.add(emb_np)

        self.metadata["ids"].extend(ids)
        self.metadata["paths"].extend(paths)
        self.metadata["splits"].extend(splits)

    def search(self, query_embeddings: torch.Tensor, top_k: int = 5):
        """Searches the GPU index and maps the resulting indices back to metadata."""
        query_np = query_embeddings.cpu().numpy().astype(np.float32)

        # similarities = inner product scores, faiss_indices = row indices
        similarities, faiss_indices = self.index.search(query_np, top_k)

        all_results = []
        for i in range(len(query_np)):
            query_res = []
            for rank in range(top_k):
                ref_idx = faiss_indices[i][rank]
                if ref_idx == -1:
                    continue  # Safety check if k > total index size

                query_res.append(
                    {
                        "score": float(similarities[i][rank]),
                        "id": self.metadata["ids"][ref_idx],
                        "split": self.metadata["splits"][ref_idx],
                        "path": self.metadata["paths"][ref_idx],
                    }
                )
            all_results.append(query_res)
        return all_results

    def save(self, output_dir: Path, prefix: str):
        """Saves the binary FAISS index and the lightweight metadata separately."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # FAISS strictly requires indices to be moved to CPU before writing to disk
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        else:
            index_to_save = self.index

        index_path = output_dir / f"{prefix}_faiss.bin"
        meta_path = output_dir / f"{prefix}_metadata.pt"

        faiss.write_index(index_to_save, str(index_path))
        torch.save(self.metadata, str(meta_path))
        print(f"Saved FAISS index to {index_path} and metadata to {meta_path}")

    @classmethod
    def load(cls, output_dir: Path, prefix: str, use_gpu: bool = True):
        """Reconstructs the FAISSEngine from disk."""
        index_path = output_dir / f"{prefix}_faiss.bin"
        meta_path = output_dir / f"{prefix}_metadata.pt"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing index or metadata for {prefix} in {output_dir}"
            )

        cpu_index = faiss.read_index(str(index_path))

        engine = cls(dimension=cpu_index.d, use_gpu=use_gpu)
        if use_gpu:
            engine.index = faiss.index_cpu_to_gpu(engine.res, 0, cpu_index)
        else:
            engine.index = cpu_index

        engine.metadata = torch.load(str(meta_path))
        return engine
