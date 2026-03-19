# EKG Similarity

PyTorch implementation of a self-supervised ECG representation learning model that combines:

- an InceptionTime-style 1D CNN for local multi-scale morphology
- a transformer encoder for long-range temporal context
- a local contrastive loss on CNN features
- a global contrastive loss on transformer embeddings
- a reconstruction decoder as a regularizer
- downstream kNN retrieval over saved global embeddings from the best checkpoint

The codebase is currently organized as a modular research scaffold. The model, losses, trainer, CSV-backed dataset loader, train/val/test dataloader setup, and embedding-based retrieval utilities are implemented. The repository also includes a smoke-test entrypoint.

## What The Code Does

Given an ECG batch, the training pipeline:

1. creates two augmented views of the same ECG
2. passes both views through the same encoder-decoder weights
3. extracts CNN-level local features and transformer-level global features
4. computes local contrastive, global contrastive, and reconstruction losses
5. saves checkpoints during training, including a per-batch checkpoint stream and `checkpoints/best.pt`
6. uses the best checkpoint to embed ECGs for downstream similarity retrieval with kNN

The intended model input shape is:

- `[batch, leads, time]`

For a 12-lead ECG recorded at 500 Hz for 10 seconds, one sample would typically be:

- `[12, 5000]`

If your data is stored as `[batch, time, leads]`, you should transpose it before the CNN.

## Getting Started

### 1. Make Sure The Environment Has PyTorch

This repository assumes you already have a Python environment with PyTorch installed.

### 2. Recommended Data Structure

The recommended repository data layout is:

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

This structure is now supported directly by the repository through the CSV-backed `ECGDataset`.

Recommended conventions:

- store one ECG per `.npy` file
- store split information in `train.csv`, `val.csv`, and `test.csv`
- keep each ECG file as either `[5000, 12]` or `[12, 5000]`
- convert each sample to `[12, 5000]` before feeding it to the model

A minimal CSV could look like:

```csv
id,path
0001,data/raw/train/sample_0001.npy
0002,data/raw/train/sample_0002.npy
```

### 3. How The Model Takes The Data

The model itself does not read folders directly. It receives tensors from a `Dataset` or `DataLoader`.

What the model expects:

- one ECG sample: `[12, 5000]`
- one batch of ECGs: `[B, 12, 5000]`

So your data flow should be:

1. load `sample_0001.npy`
2. convert it to a PyTorch tensor
3. make sure the shape is `[12, 5000]`
4. stack samples into batches of shape `[B, 12, 5000]`
5. pass the batch to the trainer

If a file is stored as `[5000, 12]`, convert it with:

```python
x = x.transpose(0, 1)  # [5000, 12] -> [12, 5000]
```

If a batch is stored as `[B, 5000, 12]`, convert it with:

```python
x = x.permute(0, 2, 1)  # [B, 5000, 12] -> [B, 12, 5000]
```

### 4. Run The Code

The current runnable command in this repository is the smoke test:

```bash
python main.py --batch-size 2 --channels 12 --sequence-length 512 --steps 1 --device cpu
```

This uses synthetic data and checks that:

- the model builds correctly
- the forward pass works
- the three losses can be computed
- backpropagation runs without shape errors

To train on the recommended folder structure, use:

```bash
python main.py --data-root data --batch-size 8 --channels 12 --sequence-length 5000 --epochs 10 --early-stopping-patience 10 --device cpu
```

With that command, the code will:

- read `data/metadata/train.csv`
- read `data/metadata/val.csv` if it exists
- read `data/metadata/test.csv` if it exists
- load the ECG files referenced in those CSV files
- build train, validation, and test dataloaders
- train for the requested number of epochs
- save a checkpoint after each training batch
- evaluate on validation and test splits when available
- track the best checkpoint as `checkpoints/best.pt` using total validation loss when available
- stop early when the total validation loss stops improving for the configured patience
- fall back to total training loss for checkpoint selection and early stopping only when no validation split exists

To run downstream ECG retrieval with kNN in the learned embedding space, use:

```bash
python retrieve.py --data-root data --reference-split train --query-split test --top-k 5 --device cpu
```

To embed all available ECGs with the best checkpoint and save a reusable retrieval index, use:

```bash
python embed_dataset.py --data-root data --splits all --batch-size 8 --channels 12 --device cpu --output-dir embeddings
```

That command writes:

- one retrieval-index file per split, for example `embeddings/train_global_index.pt`
- one combined retrieval index, for example `embeddings/all_global_index.pt`

To query against the saved combined index, use:

```bash
python retrieve.py --data-root data --reference-index embeddings/all_global_index.pt --query-split test --top-k 5 --device cpu
```

If you prefer not to save the combined index first, you can build a reference index from all available splits on the fly:

```bash
python retrieve.py --data-root data --reference-split all --query-split test --top-k 5 --device cpu
```

If you have a saved model checkpoint, you can load it with:

```bash
python retrieve.py --data-root data --reference-split train --query-split test --top-k 5 --checkpoint path/to/checkpoint.pt --device cpu
```

### 5. Prepare Your ECG Tensor Shape

The model expects:

- `[batch, leads, time]`

For your case:

- 12 leads
- 500 Hz
- 10 seconds

one sample corresponds to:

- `500 * 10 = 5000` time samples
- shape `[12, 5000]`

If your data starts as `[batch, 5000, 12]`, convert it with:

```python
x = x.permute(0, 2, 1)
```

If your data is conceptually `500 x 10 x 12`, then the first two axes are both time. In practice, that should be reshaped into a single time axis before the model, for example to `[5000, 12]` for one ECG.

### 6. Build The Default Trainer

The easiest way to start is to use `build_trainer` from `training/train.py`:

```python
from training.train import build_trainer

trainer = build_trainer(device="cpu")
```

This creates:

- the ECG model
- the combined loss function
- the optimizer
- the default augmentation pipeline

### 7. Train On A Batch

The trainer accepts a raw ECG tensor and will create the two augmented views internally:

```python
import torch
from training.train import build_trainer

trainer = build_trainer(device="cpu")

batch = torch.randn(8, 12, 5000)  # [batch, leads, time]
metrics = trainer.step(batch, train=True)
print(metrics)
```

### 8. Train From A DataLoader

If your `DataLoader` returns raw ECG tensors shaped `[B, 12, T]`, you can do:

```python
for batch in dataloader:
    metrics = trainer.step(batch, train=True)
```

If your `DataLoader` returns a dictionary, use:

```python
for batch in dataloader:
    metrics = trainer.step({"signal": batch["signal"]}, train=True)
```

### 9. When You Want To Customize Things

The main places to start changing behavior are:

- `models/encoder.py` for architecture settings
- `data/augmentations.py` for augmentation strength and type
- `losses/total_loss.py` for loss weights and temperatures
- `training/train.py` for optimizer and trainer defaults

## Repository Overview

### Entry Point

- `main.py`
  Runs a smoke test on synthetic data so you can verify that the full training stack executes.

### Data

- `data/augmentations.py`
  Defines the two-view ECG augmentation pipeline used before self-supervised training.

- `data/dataset.py`
  Defines `ECGDataset` and the train/val/test dataloader builders for the `data/raw/...` and `data/metadata/...` structure.

### Models

- `models/inception.py`
  Implements the InceptionTime-style CNN backbone.

- `models/transformer.py`
  Implements sinusoidal positional encoding and the transformer encoder.

- `models/projection_head.py`
  Legacy projection-head module retained from the earlier contrastive setup. It is not used in the current training or retrieval path.

- `models/decoder.py`
  Reconstructs the ECG from transformer tokens.

- `models/encoder.py`
  Assembles the full model and returns all outputs needed by the losses.

### Losses

- `losses/contrastive.py`
  Implements the NT-Xent contrastive loss.

- `losses/reconstruction.py`
  Implements reconstruction losses such as MSE, L1, and Smooth L1.

- `losses/total_loss.py`
  Combines local contrastive, global contrastive, and reconstruction losses into one objective.

### Training

- `training/trainer.py`
  Defines the training step and handles batch preparation, forward passes, loss computation, and optimizer updates.

- `training/train.py`
  Builds the trainer, model, optimizer, and default augmentation setup.

### Retrieval

- `utils/retrieval.py`
  Defines embedding extraction, retrieval-index creation, checkpoint loading, and kNN search utilities.

- `retrieve.py`
  Builds a searchable embedding index from one split and queries similar ECGs from another split.

- `embed_dataset.py`
  Loads the best checkpoint, embeds the selected dataset splits, and saves reusable retrieval-index files.

### Documentation

- `docs/architecture.md`
  Architecture note and Mermaid diagram for the current model design.

## Most Essential Classes And Functions

These are the most important places to start reading:

- `ECGContrastiveAutoencoder` in `models/encoder.py`
  This is the main model. It connects the CNN, transformer, and decoder, and exposes the global embedding used for retrieval.

- `InceptionEncoder` in `models/inception.py`
  This is the CNN feature extractor. It is where the local ECG morphology is modeled.

- `TemporalTransformerEncoder` in `models/transformer.py`
  This handles positional encoding and self-attention over time.

- `ECGTrainingObjective` in `losses/total_loss.py`
  This defines how the three losses are computed and combined.

- `ContrastiveAutoencoderTrainer.step` in `training/trainer.py`
  This is the core training iteration.

- `build_trainer` in `training/train.py`
  This is the easiest place to start when changing how the default training stack is built.

- `smoke_test` in `training/train.py`
  This is useful when checking whether architectural changes still run end-to-end.

- `train_with_dataloaders` in `training/train.py`
  This is the main entrypoint for training from the folder structure described in this README, including checkpoint saving.

- `extract_embeddings` and `build_retrieval_index` in `utils/retrieval.py`
  These are the core utilities for turning the trained global embeddings into searchable kNN indices.

- `embed_dataset.py`
  This is the easiest way to create a saved retrieval index from the best checkpoint after training.

## Where To Change Parameters

### Model Architecture Parameters

Change these in `ECGEncoderConfig` inside `models/encoder.py`:

- `input_channels`
  Number of ECG leads.

- `inception_depth`
  Number of Inception blocks in the CNN.

- `inception_out_channels`
  Number of filters per branch in each Inception module.

- `inception_kernel_sizes`
  Temporal kernel sizes used by the multi-scale CNN branches.

- `bottleneck_channels`
  Width of the `1x1` bottleneck before the multi-scale convolutions.

- `transformer_dim`
  Token feature dimension passed into the transformer.

- `transformer_layers`
  Number of transformer encoder layers.

- `transformer_heads`
  Number of attention heads.

- `transformer_feedforward_dim`
  Feedforward width inside each transformer block.

- `local_pool_bins`
  Number of pooled temporal bins used to form the CNN local embedding.

- `dropout`
  Dropout used in the transformer and decoder.

- `max_sequence_length`
  Maximum sequence length covered by the default positional encoding table.

### Loss Parameters

Change these in `losses/total_loss.py`:

- `LossWeights.local`
  Weight of the CNN-level contrastive loss.

- `LossWeights.global_`
  Weight of the transformer-level contrastive loss.

- `LossWeights.reconstruction`
  Weight of the reconstruction loss.

- `local_temperature`
  Temperature used by NT-Xent for the local embeddings.

- `global_temperature`
  Temperature used by NT-Xent for the global embeddings.

- `reconstruction_mode`
  Choose between `"mse"`, `"l1"`, or `"smooth_l1"`.

### Augmentation Parameters

Change these in `data/augmentations.py`:

- `RandomAmplitudeScale`
- `GaussianNoise`
- `RandomTimeShift`
- `RandomTimeMask`
- `RandomLeadDropout`
- `BaselineWander`

This is where you control augmentation probabilities and magnitudes.

### Training And Optimization Parameters

Change these in `TrainConfig` and `build_trainer` inside `training/train.py`:

- `batch_size`
- `sequence_length`
- `learning_rate`
- `weight_decay`
- `device`
- `epochs`
- `checkpoint_dir`
- `save_every_batch`
- `early_stopping_patience`
- `early_stopping_min_delta`
- optimizer choice

Set `--early-stopping-patience -1` if you want to disable early stopping from the command line.

### Retrieval Parameters

Change these in `retrieve.py` and `utils/retrieval.py`:

- `embedding_type`
  Choose whether retrieval uses the `global` or `local` embedding. `retrieval` is kept as an alias for the global embedding.

- `reference_index`
  Use a previously saved retrieval index instead of rebuilding the reference embeddings on the fly.

- `top_k`
  Number of nearest ECGs returned for each query.

- `reference_split`
  The split used to build the searchable embedding index. Use `all` to search across every available split.

- `query_split`
  The split used as retrieval queries.

## Running The Code

Run a quick smoke test with:

```bash
python main.py --batch-size 2 --channels 12 --sequence-length 512 --steps 1 --device cpu
```

This does not train on a real dataset. It only confirms that the model, losses, and trainer run end-to-end on synthetic data.

Run split-based training on real ECG files with:

```bash
python main.py --data-root data --batch-size 8 --channels 12 --sequence-length 5000 --epochs 10 --early-stopping-patience 10 --device cpu
```

Run downstream retrieval with kNN in the learned embedding space with:

```bash
python retrieve.py --data-root data --reference-split train --query-split test --top-k 5 --device cpu
```

Create reusable retrieval indices from the best checkpoint with:

```bash
python embed_dataset.py --data-root data --splits all --batch-size 8 --channels 12 --device cpu --output-dir embeddings
```

## Practical Notes

- The trainer accepts raw `signal` batches and generates augmented views internally.
- The model currently assumes that the input already has the correct shape for `Conv1d`.
- The recommended on-disk data layout is `data/raw/{train,val,test}` with matching CSV files in `data/metadata/`.
- Relative `--data-root` paths are resolved from the repository root.
- The local contrastive branch currently uses pooled CNN features rather than token-level local contrast.
- For raw `signal` batches, the reconstruction target is the original ECG, not the augmented view.
- The global contrastive loss is applied directly to the transformer global embeddings.
- There is no separately trained retrieval head in the current pipeline.
- Downstream retrieval uses the best saved checkpoint and kNN over the normalized transformer global embedding space.
- `retrieve.py` can either build a reference index on the fly or read a previously saved index from `embed_dataset.py`.
- Early stopping and best-checkpoint selection use total validation loss when a validation split is available.

## Suggested Reading Order

If you are new to the repo, the easiest reading order is:

1. `README.md`
2. `docs/architecture.md`
3. `models/encoder.py`
4. `losses/total_loss.py`
5. `training/trainer.py`
6. `data/augmentations.py`
