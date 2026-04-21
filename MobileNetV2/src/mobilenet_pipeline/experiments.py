from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import MatrixConfig, TrainConfig, dump_json
from .train import run_training


def _parse_image_size(dir_name: str) -> int:
    token = dir_name.lower().split("x")[0]
    if token.isdigit():
        return int(token)
    raise ValueError(f"Unable to parse image size from directory name: {dir_name}")


def run_experiment_matrix(config: MatrixConfig) -> dict[str, Any]:
    data_root = Path(config.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    if not config.image_size_dirs:
        config.image_size_dirs = sorted([p.name for p in data_root.iterdir() if p.is_dir()])

    all_results: list[dict[str, Any]] = []
    for size_dir in config.image_size_dirs:
        dataset_dir = data_root / size_dir
        if not dataset_dir.exists():
            continue

        image_size = _parse_image_size(size_dir)

        for augmentation in config.augment_options or [False, True]:
            run_name = f"size_{size_dir}_aug_{str(augmentation).lower()}"
            train_config = TrainConfig(
                dataset_dir=str(dataset_dir),
                image_size=image_size,
                output_dir=config.output_dir,
                run_name=run_name,
                batch_size=config.batch_size,
                epochs=config.epochs,
                seed=config.seed,
                train_split=config.train_split,
                val_split=config.val_split,
                test_split=config.test_split,
                augmentation=augmentation,
                learning_rate=config.learning_rate,
                dropout_rate=config.dropout_rate,
                base_trainable=config.base_trainable,
                early_stopping_patience=config.early_stopping_patience,
            )
            result = run_training(train_config)
            all_results.append(
                {
                    "run_name": run_name,
                    "dataset_dir": str(dataset_dir),
                    "image_size": image_size,
                    "augmentation": augmentation,
                    "test_metrics": result["test_metrics"],
                    "run_dir": result["run_dir"],
                }
            )

    payload = {
        "matrix_config": config.to_dict(),
        "num_runs": len(all_results),
        "results": all_results,
    }

    dump_json(payload, Path(config.output_dir) / "matrix_results.json")
    return payload
