from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mobilenet_pipeline.config import TrainConfig
from mobilenet_pipeline.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 transfer model on a dataset folder.")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory (class subfolders).")
    parser.add_argument("--image-size", required=True, type=int, help="Input image size for resizing (e.g., 28).")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"), help="Directory for run artifacts.")
    parser.add_argument("--run-name", default="", help="Optional run name.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--augmentation", action="store_true", help="Enable image augmentation.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--base-trainable", action="store_true", help="Unfreeze MobileNetV2 base layers.")
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        dataset_dir=args.dataset_dir,
        image_size=args.image_size,
        output_dir=args.output_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        augmentation=args.augmentation,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        base_trainable=args.base_trainable,
        early_stopping_patience=args.early_stopping_patience,
    )
    result = run_training(config)
    print(f"Training complete. Run artifacts: {result['run_dir']}")
    print(f"Test accuracy: {result['test_metrics']['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
