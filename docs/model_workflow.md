# ECG Model Workflow

This figure shows the current implementation exactly as it is used in the repository.

## Full Pipeline

```mermaid
flowchart TD
    A["Raw ECG batch<br/>shape: [B, 12, T]"] --> B["Two-view augmentation<br/>view1, view2"]
    A --> RT["Reconstruction target<br/>original ECG"]

    B --> V1["Augmented view 1<br/>[B, 12, T]"]
    B --> V2["Augmented view 2<br/>[B, 12, T]"]

    subgraph S1["Shared model weights: branch for view 1"]
        C1["InceptionTime CNN<br/>input: [B, 12, T]<br/>output: [B, 128, T]"]
        D1["Local branch<br/>AdaptiveAvgPool1d(8)<br/>flatten -> h1 [B, 1024]"]
        E1["Transpose CNN output<br/>[B, 128, T] -> [B, T, 128]"]
        F1["Add sinusoidal positional encoding"]
        G1["Transformer encoder<br/>tokens: [B, T, 128]"]
        H1["Mean over time + LayerNorm<br/>global embedding g1 [B, 128]"]
        I1["Decoder<br/>reconstruction r1 [B, 12, T]"]
    end

    subgraph S2["Shared model weights: branch for view 2"]
        C2["InceptionTime CNN<br/>input: [B, 12, T]<br/>output: [B, 128, T]"]
        D2["Local branch<br/>AdaptiveAvgPool1d(8)<br/>flatten -> h2 [B, 1024]"]
        E2["Transpose CNN output<br/>[B, 128, T] -> [B, T, 128]"]
        F2["Add sinusoidal positional encoding"]
        G2["Transformer encoder<br/>tokens: [B, T, 128]"]
        H2["Mean over time + LayerNorm<br/>global embedding g2 [B, 128]"]
        I2["Decoder<br/>reconstruction r2 [B, 12, T]"]
    end

    V1 --> C1
    C1 --> D1
    C1 --> E1 --> F1 --> G1 --> H1
    G1 --> I1

    V2 --> C2
    C2 --> D2
    C2 --> E2 --> F2 --> G2 --> H2
    G2 --> I2

    D1 -. "Local NT-Xent loss" .- D2
    H1 -. "Global NT-Xent loss" .- H2
    I1 -. "Reconstruction loss vs original ECG" .- RT
    I2 -. "Reconstruction loss vs original ECG" .- RT

    subgraph O["Total training objective"]
        L["Total loss =<br/>w_local * local_loss +<br/>w_global * global_loss +<br/>w_recon * reconstruction_loss"]
    end

    D1 --> L
    D2 --> L
    H1 --> L
    H2 --> L
    I1 --> L
    I2 --> L

    L --> CK["Save checkpoint after each training batch"]
    CK --> BEST["Select best checkpoint<br/>lowest total val loss<br/>fallback: total train loss if no val split"]

    BEST --> EMB["Load best checkpoint<br/>embed ECG database"]

    subgraph R["Downstream retrieval"]
        DB["All ECGs in dataset<br/>[B, 12, T]"] --> RCNN["CNN -> Transformer"]
        RCNN --> RG["Global embedding g [B, 128]"]
        RG --> RN["L2 normalize embedding"]
        RN --> IDX["Saved embedding index"]
        Q["Query ECG"] --> QCNN["Same encoder"]
        QCNN --> QG["Query global embedding"]
        QG --> QN["L2 normalize query embedding"]
        QN --> KNN["Cosine similarity / kNN search"]
        IDX --> KNN
        KNN --> RES["Most similar ECGs"]
    end

    EMB --> DB
    EMB --> Q
```

## Short Reading Guide

- The CNN learns local ECG morphology from each augmented view.
- The pooled CNN features `h1` and `h2` are compared with the local contrastive loss.
- The transformer receives the CNN features after transpose and positional encoding.
- The mean-pooled transformer output `g1` and `g2` is used for the global contrastive loss.
- The decoder reconstructs the original ECG, not the augmented view.
- The best checkpoint is chosen by total validation loss.
- After training, the best checkpoint is used to create normalized global embeddings for retrieval.
- Retrieval is kNN or cosine search in that embedding space.
