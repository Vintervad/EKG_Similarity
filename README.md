# EKG Similarity

Self-supervised ECG representation learning for similarity retrieval.

The repository trains an InceptionTime-style CNN + Transformer encoder on raw ECGs, saves checkpoints during training, selects the best checkpoint by validation loss, embeds the ECG database, and retrieves the most similar ECGs with cosine similarity / kNN.

Retrieval now supports two backends:

- `torch`:
  exact cosine similarity / brute-force kNN over saved embeddings
- `faiss`:
  FAISS-based vector search, including approximate nearest neighbor (ANN) indices such as IVF

## Overview

The current pipeline is:

1. load ECGs from `train`, `val`, and `test`
2. create two augmented views of each ECG
3. encode both views with shared weights
4. apply local contrastive loss on pooled CNN features
5. apply global contrastive loss on a projection head fed by the transformer global embedding
6. reconstruct the original ECG from transformer tokens
7. save checkpoints during training and track `checkpoints/best.pt`
8. use the best checkpoint to embed the ECG database
9. retrieve similar ECGs from the embedding space with exact kNN or FAISS ANN

The important design choice is:

- global contrastive training uses the projection head
- downstream retrieval uses the pre-head transformer global embedding

## Python Dependencies

### Local environment checked here

The code has been checked in this local development environment with:

- `Python 3.12.7`
- `torch 2.10.0+cu126`
- `numpy 1.26.4`
- `wfdb 4.3.1`
- `faiss-cpu 1.13.2`

The direct Python packages used by the repository are:

| Package | Version used here | Needed for |
|---|---:|---|
| `torch` | `2.10.0+cu126` | training, embedding, retrieval |
| `numpy` | `1.26.4` | ECG loading and tensor preparation |
| `wfdb` | `4.3.1` | dataset preparation from WFDB/PTB-XL style records |
| `faiss-cpu` | `1.13.2` | FAISS retrieval backend on CPU in local development |

Notes:

- If you want to use the FAISS backend on a GPU-enabled HPC system, install a **GPU-enabled FAISS build** instead of `faiss-cpu`.
- The core training code does **not** require FAISS unless you use `--backend faiss`.
- The core training and retrieval code does **not** require `wfdb` unless you run [data/prepare_ptbxl.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/data/prepare_ptbxl.py).

Example installs for the environment used here:

```bash
python -m pip install numpy==1.26.4
python -m pip install wfdb==4.3.1
python -m pip install faiss-cpu==1.13.2
```

For PyTorch, install the build that matches your CUDA setup. The code here was checked with:

```text
torch 2.10.0+cu126
```

### HPC GPU FAISS setup

The codebase now supports **FAISS GPU in the retrieval code**, but that requires a GPU-enabled FAISS installation on the HPC.

Important distinction:

- **local machine used here:** `faiss-cpu`
- **HPC target for large-scale ANN search:** GPU-enabled FAISS

So if you run:

```bash
--backend faiss --faiss-use-gpu
```

then the Python environment on the HPC must expose FAISS GPU bindings. The code will try to use:

- `faiss.StandardGpuResources()`
- `faiss.index_cpu_to_gpu(...)`

If those symbols are not available, the script will fail with a clear error telling you that the installed FAISS build does not support GPU.

#### What to install on the HPC

On the HPC, you need:

- a Python environment with your project packages
- a CUDA-compatible PyTorch build
- a **GPU-enabled FAISS build**

The exact installation command depends on how the cluster manages software:

- a cluster module
- a conda environment
- or a prebuilt shared environment from your institution

Because HPC environments differ, the most reliable recommendation is:

1. load the cluster CUDA/Python modules required by your site
2. create or activate the project environment
3. install a **GPU-enabled FAISS package** for that environment
4. verify that this works:

```bash
python -c "import faiss; print(hasattr(faiss, 'StandardGpuResources'))"
```

That command should print:

```text
True
```

If it prints `False`, then FAISS is installed, but not the GPU-enabled build.

#### Recommended HPC verification

Before building a 12M-vector index, verify all three checks:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import faiss; print(faiss.__version__)"
python -c "import faiss; print(hasattr(faiss, 'StandardGpuResources'))"
```

Expected:

- PyTorch CUDA available: `True`
- FAISS imports successfully
- GPU FAISS support available: `True`

#### Example HPC FAISS command

Once GPU FAISS is installed on the cluster, a typical ANN index build command is:

```bash
python embed_dataset.py --data-root data --splits all --batch-size 8 --channels 12 --device cuda --output-dir embeddings --backend faiss --faiss-index-type ivf-flat --faiss-use-gpu --faiss-gpu-device 0 --faiss-nlist 65536 --faiss-nprobe 64 --faiss-train-size 200000
```

And query:

```bash
python retrieve.py --data-root data --reference-index embeddings/all_global_index.faiss --query-split test --top-k 5 --device cuda --backend faiss --faiss-use-gpu --faiss-gpu-device 0 --faiss-nprobe 64
```

#### Example Slurm job for FAISS GPU indexing

There is also a ready-to-edit Slurm template here:

- [slurm/build_faiss_index.sbatch](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/slurm/build_faiss_index.sbatch)

It assumes:

- you already trained the model
- you have a `best.pt` checkpoint
- your HPC environment provides CUDA and a GPU-enabled FAISS build

Submit it with:

```bash
sbatch slurm/build_faiss_index.sbatch
```

The template does three things:

1. checks that PyTorch sees the GPU
2. checks that the FAISS build exposes `StandardGpuResources`
3. builds a FAISS IVF index from the trained ECG embeddings

## Data Format

The model expects ECG tensors in:

- `[batch, leads, time]`

For a 12-lead ECG sampled at 500 Hz for 10 seconds, one sample should usually be:

- `[12, 5000]`

If your raw file is `[5000, 12]`, transpose it before the model:

```python
x = x.transpose(0, 1)
```

If a batch is `[B, 5000, 12]`, convert it with:

```python
x = x.permute(0, 2, 1)
```

## Recommended Dataset Structure

```text
data/
  raw/
    train/
      sample_0001.npy
      sample_0002.npy
    val/
      sample_1001.npy
    test/
      sample_2001.npy
  metadata/
    train.csv
    val.csv
    test.csv
```

Each CSV should contain at least:

```csv
id,path
0001,data/raw/train/sample_0001.npy
0002,data/raw/train/sample_0002.npy
```

Supported ECG file formats:

- `.npy`
- `.pt`
- `.pth`

The dataset loader can handle shapes like:

- `[12, 5000]`
- `[5000, 12]`
- `[500, 10, 12]`

as long as exactly one axis is the lead axis of size `12`.

## Quick Start

### 1. Smoke Test

Use this to verify that the model, losses, and trainer run end-to-end:

```bash
python main.py --batch-size 2 --channels 12 --sequence-length 512 --steps 1 --device cpu
```

### 2. Train The Model

CPU:

```bash
python main.py --data-root data --batch-size 8 --channels 12 --sequence-length 5000 --epochs 50 --early-stopping-patience 10 --device cpu
```

GPU:

```bash
python main.py --data-root data --batch-size 8 --channels 12 --sequence-length 5000 --epochs 50 --early-stopping-patience 10 --device cuda
```

This will:

- load `train.csv`, and optionally `val.csv` and `test.csv`
- save one checkpoint after every training epoch
- keep per-epoch checkpoints in `checkpoints/epochs/`
- keep `checkpoints/latest.pt`
- keep `checkpoints/best.pt`
- write `checkpoints/metrics/train_batch_metrics.csv`
- write `checkpoints/metrics/epoch_metrics.csv`
- use total validation loss for best-checkpoint selection when `val.csv` exists
- use early stopping on total validation loss when `val.csv` exists

If there is no validation split, it falls back to total training loss.

Set `--early-stopping-patience -1` to disable early stopping.

### 3. Augmentation Modes

The repository supports two modes for creating the two augmented views used in contrastive training:

#### Default Mode (`--augment-mode default`)
Uses synthetic transforms:
- Random Amplitude Scaling
- Gaussian Noise
- Random Time Shifting
- Random Time Masking
- Baseline Wander (Sinusoidal)

#### PhysioNet Mode (`--augment-mode physionet`)
Uses real-world noise from the PhysioNet NSTDB database:
1. **View 1 (Clean)**: Applies a clinical 0.016-150Hz bandpass filter to the original ECG.
2. **View 2 (Noisy)**: Applies the same clinical filter, then adds a combination of Muscle Artifact (`ma`), Baseline Wander (`bw`), and Electrode Motion (`em`) from the noise banks.

To use this mode, ensure the noise files exist (run `python preproc/setup_noise.py`) and specify the directory:

```bash
python main.py --data-root data --augment-mode physionet --physionet-noise-dir physionet_data --physionet-target-snr 5.0
```

### 4. Embed The ECG Database

After training, create retrieval indices from the best checkpoint.

Exact torch index:

```bash
python embed_dataset.py --data-root data --splits all --batch-size 8 --channels 12 --device cuda --output-dir embeddings
```

This creates:

- per-split indices such as `embeddings/train_global_index.pt`
- a combined index such as `embeddings/all_global_index.pt`

Approximate FAISS IVF index:

```bash
python embed_dataset.py --data-root data --splits all --batch-size 8 --channels 12 --device cuda --output-dir embeddings --backend faiss --faiss-index-type ivf-flat --faiss-use-gpu --faiss-nlist 65536 --faiss-nprobe 64 --faiss-train-size 200000
```

This creates:

- per-split FAISS indices such as `embeddings/train_global_index.faiss`
- per-split metadata such as `embeddings/train_global_index_meta.pt`
- a combined FAISS index such as `embeddings/all_global_index.faiss`
- combined metadata such as `embeddings/all_global_index_meta.pt`

The CSV logs can be used directly for graphs in pandas, matplotlib, Excel, or similar tools.

### 4. Retrieve Similar ECGs

Query against the saved combined index:

```bash
python retrieve.py --data-root data --reference-index embeddings/all_global_index.pt --query-split test --top-k 5 --device cuda
```

Or build the reference index on the fly from all available splits:

```bash
python retrieve.py --data-root data --reference-split all --query-split test --top-k 5 --device cuda
```

Query a saved FAISS ANN index:

```bash
python retrieve.py --data-root data --reference-index embeddings/all_global_index.faiss --query-split test --top-k 5 --device cuda --backend faiss --faiss-use-gpu --faiss-nprobe 64
```

Or build a FAISS ANN index on the fly from all available splits:

```bash
python retrieve.py --data-root data --reference-split all --query-split test --top-k 5 --device cuda --backend faiss --faiss-index-type ivf-flat --faiss-use-gpu --faiss-nlist 65536 --faiss-nprobe 64 --faiss-train-size 200000
```

## FAISS Backend

FAISS is used to speed up retrieval in large embedding databases.

In this project, the encoder produces one normalized embedding vector per ECG, typically:

- `[N, 128]`

where `N` is the number of ECGs and `128` is the embedding dimension.

The FAISS backend does the following:

1. extracts normalized retrieval embeddings from the trained encoder
2. trains an optional ANN index if the chosen index family requires training
3. adds all ECG embeddings to the FAISS index
4. stores the FAISS index separately from the ECG metadata
5. searches the index for the nearest neighbors of each query ECG

### Which FAISS index to use

- `flat`:
  exact search, no FAISS training step, slowest at very large scale
- `ivf-flat`:
  approximate nearest neighbor search using an inverted file index; recommended first ANN option for this project
- `ivf-pq`:
  approximate search with product quantization compression; useful later if memory becomes the limiting factor

### Why `ivf-flat` is the default FAISS option

`ivf-flat` gives a good tradeoff for a first large-scale retrieval system:

- much faster than brute-force search
- less approximation error than `ivf-pq`
- still compatible with cosine-style retrieval through inner-product search on normalized embeddings

### Important FAISS files

- `.faiss`:
  the vector index itself
- `_meta.pt`:
  the metadata needed to map FAISS result indices back to ECG ids, file paths, and dataset splits

### FAISS dependency

FAISS is optional. The default retrieval backend remains `torch`.

To use `--backend faiss`, install a FAISS build that matches your machine:

- CPU-only FAISS, or
- GPU-enabled FAISS if you want `--faiss-use-gpu`

If FAISS is not installed, the scripts will fail with a clear import error when you request the FAISS backend.

## Model Summary

The model lives mainly in [models/encoder.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/models/encoder.py).

It does the following:

- CNN backbone:
  InceptionTime-style 1D convolutions extract local ECG morphology across all leads.
- Local branch:
  pooled CNN features are used for the local NT-Xent loss.
- Transformer branch:
  CNN features are transposed to `[B, T, D]`, positional encoding is added, and self-attention models long-range temporal structure.
- Global embedding:
  transformer tokens are mean-pooled and normalized to produce the retrieval embedding.
- Projection head:
  the global embedding is passed through a projection head for the global NT-Xent loss.
- Decoder:
  transformer tokens are decoded to reconstruct the original ECG.

In short:

- local loss trains CNN morphology features
- global loss trains the projection head on top of the transformer embedding
- retrieval uses the pre-head transformer global embedding

## Main Files

- [main.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/main.py)
  Command-line entrypoint for smoke tests and dataset training.

- [data/dataset.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/data/dataset.py)
  CSV-backed `ECGDataset` and train/val/test dataloader builders.

- [data/augmentations.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/data/augmentations.py)
  Two-view ECG augmentations used for self-supervised training.

- [models/inception.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/models/inception.py)
  InceptionTime-style CNN encoder.

- [models/transformer.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/models/transformer.py)
  Positional encoding and transformer encoder.

- [models/projection_head.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/models/projection_head.py)
  Projection head used for the global contrastive branch.

- [models/decoder.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/models/decoder.py)
  Reconstruction decoder.

- [losses/total_loss.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/losses/total_loss.py)
  Combined local contrastive, global contrastive, and reconstruction objective.

- [training/trainer.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/training/trainer.py)
  One training step, batch preparation, forward pass, and optimizer step.

- [training/train.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/training/train.py)
  Trainer construction, dataloader training loop, checkpointing, best-checkpoint selection, and early stopping.

- [embed_dataset.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/embed_dataset.py)
  Builds reusable retrieval indices from the best checkpoint for either the torch or FAISS backend.

- [retrieve.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/retrieve.py)
  Queries the embedding index to find similar ECGs with either exact kNN or FAISS ANN.

- [utils/faiss_retrieval.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/utils/faiss_retrieval.py)
  FAISS ANN index building, saving/loading, and querying.

- [docs/model_workflow.md](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/docs/model_workflow.md)
  Standalone figure of the full train-to-retrieval workflow.

## Most Important Parameters

Change these in [models/encoder.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/models/encoder.py):

- `input_channels`
- `inception_depth`
- `inception_out_channels`
- `inception_kernel_sizes`
- `bottleneck_channels`
- `transformer_dim`
- `transformer_layers`
- `transformer_heads`
- `transformer_feedforward_dim`
- `projection_head_hidden_dim`
- `projection_head_output_dim`
- `local_pool_bins`
- `dropout`
- `max_sequence_length`

Change these in [losses/total_loss.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/losses/total_loss.py):

- `LossWeights.local`
- `LossWeights.global_`
- `LossWeights.reconstruction`
- `local_temperature`
- `global_temperature`
- `reconstruction_mode`

Change these in [training/train.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/training/train.py):

- `batch_size`
- `learning_rate`
- `weight_decay`
- `epochs`
- `checkpoint_dir`
- `save_every_epoch`
- `early_stopping_patience`
- `early_stopping_min_delta`

Change these in [retrieve.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/retrieve.py):

- `backend`
- `reference_split`
- `reference_index`
- `query_split`
- `top_k`
- `embedding_type`

When using FAISS, the most important extra CLI parameters are:

- `faiss_index_type`
- `faiss_use_gpu`
- `faiss_nlist`
- `faiss_nprobe`
- `faiss_train_size`
- `faiss_pq_m`
- `faiss_pq_bits`

## Practical Notes

- GPU works if your PyTorch installation supports CUDA. Use `--device cuda`.
- FAISS GPU support is separate from PyTorch CUDA support. `--faiss-use-gpu` only works with a GPU-enabled FAISS build.
- Reconstruction is done from the full transformer token sequence, not from the pooled global embedding.
- Retrieval uses the normalized pre-head transformer global embedding by default.
- The FAISS backend also uses the normalized pre-head transformer global embedding by default.
- If the query ECG is also present in the reference index, it can retrieve itself as the nearest neighbor.
- `flat` FAISS indices are exact; `ivf-flat` and `ivf-pq` are approximate and trade some recall for speed.
- Because the architecture now includes a projection head, train fresh checkpoints instead of reusing checkpoints from the older no-head version.
- Training loss is written both per batch and per epoch under `checkpoints/metrics/`.
- Validation loss is written only per epoch in `checkpoints/metrics/epoch_metrics.csv`.

## Suggested Reading Order

1. [docs/model_workflow.md](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/docs/model_workflow.md)
2. [models/encoder.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/models/encoder.py)
3. [losses/total_loss.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/losses/total_loss.py)
4. [training/train.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/training/train.py)
5. [retrieve.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/retrieve.py)
