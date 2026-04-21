from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mobilenet_pipeline.config import MatrixConfig
from mobilenet_pipeline.experiments import run_experiment_matrix


def _parse_bool(value: str) -> bool:
    lowered = value.lower().strip()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean from: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix across sizes and augmentation settings.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT), help="Root folder containing size directories (e.g., 28x28).")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"), help="Directory for matrix and run outputs.")
    parser.add_argument(
        "--size-dirs",
        nargs="+",
        default=None,
        help="Image size folder names under data root (example: 28x28 56x56).",
    )
    parser.add_argument(
        "--augment-options",
        nargs="+",
        default=["false", "true"],
        help="Booleans for augmentation options (example: false true).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--base-trainable", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    args = parser.parse_args()

    augment_options = [_parse_bool(v) for v in args.augment_options]

    config = MatrixConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        image_size_dirs=args.size_dirs,
        augment_options=augment_options,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        base_trainable=args.base_trainable,
        early_stopping_patience=args.early_stopping_patience,
    )

    result = run_experiment_matrix(config)
    print(f"Matrix complete. Runs executed: {result['num_runs']}")
    print(f"Results file: {Path(args.output_dir) / 'matrix_results.json'}")


if __name__ == "__main__":
    main()
