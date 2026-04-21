from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

import tensorflow as tf

from .config import TrainConfig, dump_json
from .data import create_datasets
from .model import build_model


def _make_run_dir(output_dir: str, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_name = run_name if run_name else f"run_{timestamp}"
    run_dir = Path(output_dir) / resolved_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _history_to_json(history: tf.keras.callbacks.History) -> dict[str, Any]:
    return {k: [float(v) for v in values] for k, values in history.history.items()}


def run_training(config: TrainConfig) -> dict[str, Any]:
    config.validate()
    tf.keras.utils.set_random_seed(config.seed)

    run_dir = _make_run_dir(config.output_dir, config.run_name)

    data_bundle = create_datasets(
        dataset_dir=config.dataset_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed,
    )

    model = build_model(
        image_size=config.image_size,
        num_classes=len(data_bundle.class_names),
        learning_rate=config.learning_rate,
        dropout_rate=config.dropout_rate,
        use_augmentation=config.augmentation,
        base_trainable=config.base_trainable,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.CSVLogger(str(run_dir / "metrics.csv")),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best.weights.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "checkpoints" / "epoch_{epoch:03d}.keras"),
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, config.early_stopping_patience // 2),
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        data_bundle.train_ds,
        validation_data=data_bundle.val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_weights = run_dir / "best.weights.h5"
    if best_weights.exists():
        model.load_weights(best_weights)

    test_loss, test_accuracy = model.evaluate(data_bundle.test_ds, verbose=0)

    model.save(run_dir / "final_model.keras")

    payload = {
        "config": config.to_dict(),
        "run_dir": str(run_dir),
        "class_names": data_bundle.class_names,
        "class_to_index": data_bundle.class_to_index,
        "split_counts": data_bundle.split_counts,
        "history": _history_to_json(history),
        "test_metrics": {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
        },
    }

    dump_json(payload, run_dir / "run_summary.json")
    dump_json(config.to_dict(), run_dir / "config.json")

    return payload


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
