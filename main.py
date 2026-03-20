from __future__ import annotations

import argparse

from training.train import TrainConfig, smoke_test, train_with_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a smoke test or train/evaluate the ECG InceptionTime + Transformer contrastive autoencoder."
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        input_channels=args.channels,
        sequence_length=args.sequence_length,
        device=args.device,
        epochs=args.epochs,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        steps=args.steps,
    )
    if args.data_root:
        results = train_with_dataloaders(config)
        for epoch_result in results["history"]:
            train_metrics = epoch_result["train"]
            train_formatted = ", ".join(f"train_{key}={value:.4f}" for key, value in train_metrics.items())
            message = f"epoch={epoch_result['epoch']}, {train_formatted}"
            if "val" in epoch_result:
                val_metrics = epoch_result["val"]
                val_formatted = ", ".join(f"val_{key}={value:.4f}" for key, value in val_metrics.items())
                message = f"{message}, {val_formatted}"
            print(message)
        if "test" in results:
            test_formatted = ", ".join(f"test_{key}={value:.4f}" for key, value in results["test"].items())
            print(test_formatted)
        print(f"splits_loaded={','.join(results['available_splits'])}")
        if results.get("best_checkpoint"):
            print(f"best_checkpoint={results['best_checkpoint']}")
        print(f"checkpoint_dir={results['checkpoint_dir']}")
        print(f"train_batch_metrics={results['train_batch_metrics_path']}")
        print(f"epoch_metrics={results['epoch_metrics_path']}")
        print(f"selection_metric={results['selection_metric_name']}")
        print(f"early_stopping_patience={results['early_stopping_patience']}")
        print(f"early_stopping_min_delta={results['early_stopping_min_delta']}")
        print(f"stopped_early={results['stopped_early']}")
        if results.get("stopped_epoch") is not None:
            print(f"stopped_epoch={results['stopped_epoch']}")
        return

    metrics_history = smoke_test(config)
    for step, metrics in enumerate(metrics_history, start=1):
        formatted = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"step={step}, {formatted}")


if __name__ == "__main__":
    main()
