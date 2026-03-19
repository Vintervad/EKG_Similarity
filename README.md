# EKG Similarity

PyTorch implementation of a self-supervised ECG representation learning model that combines:

- an InceptionTime-style 1D CNN for local multi-scale morphology
- a transformer encoder for long-range temporal context
- a local contrastive loss on CNN features
- a global contrastive loss on transformer features
- a reconstruction decoder as a regularizer

The codebase is currently organized as a modular research scaffold. The model, losses, and trainer are implemented, and the repository includes a smoke-test entrypoint. Dataset-specific loading and experiment management are not yet built out.

## What The Code Does

Given an ECG batch, the training pipeline:

1. creates two augmented views of the same ECG
2. passes both views through the same encoder-decoder weights
3. extracts CNN-level local features and transformer-level global features
4. computes local contrastive, global contrastive, and reconstruction losses
5. combines those losses into a single objective for training

The intended model input shape is:

- `[batch, leads, time]`

For a 12-lead ECG recorded at 500 Hz for 10 seconds, one sample would typically be:

- `[12, 5000]`

If your data is stored as `[batch, time, leads]`, you should transpose it before the CNN.

## Repository Overview

### Entry Point

- `main.py`
  Runs a smoke test on synthetic data so you can verify that the full training stack executes.

### Data

- `data/augmentations.py`
  Defines the two-view ECG augmentation pipeline used before self-supervised training.

### Models

- `models/inception.py`
  Implements the InceptionTime-style CNN backbone.

- `models/transformer.py`
  Implements sinusoidal positional encoding and the transformer encoder.

- `models/projection_head.py`
  Maps the transformer global embedding into the contrastive embedding space.

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

### Documentation

- `docs/architecture.md`
  Architecture note and Mermaid diagram for the current model design.

## Most Essential Classes And Functions

These are the most important places to start reading:

- `ECGContrastiveAutoencoder` in `models/encoder.py`
  This is the main model. It connects the CNN, transformer, projection head, and decoder.

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

- `projection_dim`
  Size of the final contrastive embedding.

- `projection_hidden_dim`
  Hidden size inside the projection head.

- `local_pool_bins`
  Number of pooled temporal bins used to form the CNN local embedding.

- `dropout`
  Dropout used in the transformer, projection head, and decoder.

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
- optimizer choice

## Running The Code

Run a quick smoke test with:

```bash
python main.py --batch-size 2 --channels 12 --sequence-length 512 --steps 1 --device cpu
```

This does not train on a real dataset. It only confirms that the model, losses, and trainer run end-to-end on synthetic data.

## Practical Notes

- The trainer accepts raw `signal` batches and generates augmented views internally.
- The model currently assumes that the input already has the correct shape for `Conv1d`.
- The local contrastive branch currently uses pooled CNN features rather than token-level local contrast.
- The reconstruction target defaults to the same augmented input view.

## Suggested Reading Order

If you are new to the repo, the easiest reading order is:

1. `README.md`
2. `docs/architecture.md`
3. `models/encoder.py`
4. `losses/total_loss.py`
5. `training/trainer.py`
6. `data/augmentations.py`
