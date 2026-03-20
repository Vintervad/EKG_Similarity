# EKG Similarity

Self-supervised ECG representation learning for similarity retrieval.

The repository trains an InceptionTime-style CNN + Transformer encoder on raw ECGs, saves checkpoints during training, selects the best checkpoint by validation loss, embeds the ECG database, and retrieves the most similar ECGs with cosine similarity / kNN.

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
9. retrieve similar ECGs from the embedding space

The important design choice is:

- global contrastive training uses the projection head
- downstream retrieval uses the pre-head transformer global embedding

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
- save a checkpoint after every training batch
- keep `checkpoints/latest.pt`
- keep `checkpoints/best.pt`
- write `checkpoints/metrics/train_batch_metrics.csv`
- write `checkpoints/metrics/epoch_metrics.csv`
- use total validation loss for best-checkpoint selection when `val.csv` exists
- use early stopping on total validation loss when `val.csv` exists

If there is no validation split, it falls back to total training loss.

Set `--early-stopping-patience -1` to disable early stopping.

### 3. Embed The ECG Database

After training, create retrieval indices from the best checkpoint:

```bash
python embed_dataset.py --data-root data --splits all --batch-size 8 --channels 12 --device cuda --output-dir embeddings
```

This creates:

- per-split indices such as `embeddings/train_global_index.pt`
- a combined index such as `embeddings/all_global_index.pt`

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
  Builds reusable retrieval indices from the best checkpoint.

- [retrieve.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/retrieve.py)
  Queries the embedding index to find similar ECGs.

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
- `save_every_batch`
- `early_stopping_patience`
- `early_stopping_min_delta`

Change these in [retrieve.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/retrieve.py):

- `reference_split`
- `reference_index`
- `query_split`
- `top_k`
- `embedding_type`

## Practical Notes

- GPU works if your PyTorch installation supports CUDA. Use `--device cuda`.
- Reconstruction is done from the full transformer token sequence, not from the pooled global embedding.
- Retrieval uses the normalized pre-head transformer global embedding by default.
- If the query ECG is also present in the reference index, it can retrieve itself as the nearest neighbor.
- Because the architecture now includes a projection head, train fresh checkpoints instead of reusing checkpoints from the older no-head version.
- Training loss is written both per batch and per epoch under `checkpoints/metrics/`.
- Validation loss is written only per epoch in `checkpoints/metrics/epoch_metrics.csv`.

## Suggested Reading Order

1. [docs/model_workflow.md](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/docs/model_workflow.md)
2. [models/encoder.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/models/encoder.py)
3. [losses/total_loss.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/losses/total_loss.py)
4. [training/train.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/training/train.py)
5. [retrieve.py](C:/Users/sebas/OneDrive/Dokumenter/Uni-Sunhedsteknologi-Kandiddat/10.semester/Git_code/EKG_Similarity/retrieve.py)
