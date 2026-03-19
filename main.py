from __future__ import annotations

import argparse

from training.train import TrainConfig, smoke_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the ECG InceptionTime + Transformer contrastive autoencoder."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=2500)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_history = smoke_test(
        TrainConfig(
            batch_size=args.batch_size,
            input_channels=args.channels,
            sequence_length=args.sequence_length,
            device=args.device,
            steps=args.steps,
        )
    )
    for step, metrics in enumerate(metrics_history, start=1):
        formatted = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"step={step}, {formatted}")


if __name__ == "__main__":
    main()
