# ECG Contrastive Autoencoder Architecture

This note describes the current implementation of the ECG encoder-decoder model and the tensor shapes used throughout the forward pass.

The current training pipeline is:

- raw ECG
- augmentation into two views
- shared encoder-decoder
- local, global, and reconstruction losses

## Diagram

```mermaid
flowchart TD
    A["Raw ECG batch<br/>recommended shape: [B, 12, T]"] --> B["Two-view augmentation<br/>view1, view2"]

    B --> V1["View 1<br/>[B, 12, T]"]
    B --> V2["View 2<br/>[B, 12, T]"]

    subgraph M1["Shared Encoder Weights"]
        C1["InceptionTime CNN<br/>multi-scale Conv1d kernels<br/>39, 19, 9 + pool branch<br/>output: [B, 128, T]"]
        D1["Local branch<br/>AdaptiveAvgPool1d(8)<br/>flatten -> h1 [B, 1024]"]
        E1["Transpose -> [B, T, 128]"]
        F1["Sinusoidal positional encoding"]
        G1["Transformer encoder<br/>output tokens: [B, T, 128]"]
        H1["Mean over time<br/>global embedding g1 [B, 128]"]
        I1["Projection head<br/>z1 [B, 128]"]
        J1["Decoder<br/>reconstruction r1 [B, 12, T]"]
    end

    subgraph M2["Shared Encoder Weights"]
        C2["InceptionTime CNN<br/>output: [B, 128, T]"]
        D2["Local branch<br/>AdaptiveAvgPool1d(8)<br/>flatten -> h2 [B, 1024]"]
        E2["Transpose -> [B, T, 128]"]
        F2["Sinusoidal positional encoding"]
        G2["Transformer encoder<br/>output tokens: [B, T, 128]"]
        H2["Mean over time<br/>global embedding g2 [B, 128]"]
        I2["Projection head<br/>z2 [B, 128]"]
        J2["Decoder<br/>reconstruction r2 [B, 12, T]"]
    end

    V1 --> C1
    C1 --> D1
    C1 --> E1
    E1 --> F1 --> G1
    G1 --> H1 --> I1
    G1 --> J1

    V2 --> C2
    C2 --> D2
    C2 --> E2
    E2 --> F2 --> G2
    G2 --> H2 --> I2
    G2 --> J2

    D1 -. local contrastive loss .- D2
    I1 -. global contrastive loss .- I2
    J1 -. reconstruction loss vs view1 .- V1
    J2 -. reconstruction loss vs view2 .- V2
```

## Tensor Shapes

The current CNN expects tensors in PyTorch `Conv1d` format:

- `[batch, channels, length]`
- here, `channels = leads`
- here, `length = time samples`

For a 12-lead ECG recorded at 500 Hz for 10 seconds:

- sampling rate = `500 Hz`
- duration = `10 s`
- total time samples = `500 * 10 = 5000`
- number of leads = `12`

So one ECG example should typically be represented as:

- `[12, 5000]` for a single sample
- `[B, 12, 5000]` for a batch

If your data is currently stored conceptually as:

- `500 x 10 x 12`

then the first two dimensions are both parts of time, not two independent model axes. In practice, that should usually be reshaped into:

- `5000 x 12` for one ECG
- or `[B, 5000, 12]` for a batch if you prefer time-major storage before feeding the CNN

## Why The Transpose Is Needed

The transpose between the CNN and transformer exists because those two modules expect different tensor layouts.

### CNN input and output

`nn.Conv1d` expects:

- `[B, C, T]`

where:

- `B` is batch size
- `C` is channels
- `T` is sequence length

In this model, the lead dimension is treated as the channel dimension, so the CNN operates on:

- input: `[B, 12, T]`
- output: `[B, 128, T]`

That means the CNN is learning temporal filters over the ECG while mixing information across leads through the convolution weights.

### Transformer input

The transformer in this repo is created with `batch_first=True`, so it expects:

- `[B, T, D]`

where:

- `T` is the sequence length
- `D` is the per-token feature dimension

After the CNN, the tensor is `[B, 128, T]`. Here:

- `128` is the feature dimension produced by the CNN
- `T` is still the time axis

To feed this into the transformer, we swap those last two axes:

- before transpose: `[B, 128, T]`
- after transpose: `[B, T, 128]`

Now each time step becomes one token, and each token has a 128-dimensional feature vector.

## In Short

- The CNN wants `leads` in the channel position.
- The transformer wants `time` in the sequence position.
- So the CNN uses `[B, 12, T]`.
- The transformer uses `[B, T, 128]`.
- The transpose is what converts from the CNN layout to the transformer layout.

## Recommended Data Convention

To keep the code simple, the cleanest convention is:

- store raw batched ECGs as `[B, T, 12]` if that feels natural for your data pipeline
- convert to `[B, 12, T]` immediately before the CNN

If your dataset already outputs `[B, 12, T]`, then no extra transpose is needed before the CNN.
