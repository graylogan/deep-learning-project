from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .config import TrainConfig, dump_json
from .data import create_datasets
from .model import build_model
from .train import load_json


def _build_config_from_run(run_dir: Path) -> TrainConfig:
    config_dict = load_json(run_dir / "config.json")
    return TrainConfig(**config_dict)


def evaluate_run(run_dir: str) -> dict[str, Any]:
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    config = _build_config_from_run(run_path)

    bundle = create_datasets(
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
        num_classes=len(bundle.class_names),
        learning_rate=config.learning_rate,
        dropout_rate=config.dropout_rate,
        use_augmentation=config.augmentation,
        base_trainable=config.base_trainable,
    )
    model.load_weights(run_path / "best.weights.h5")

    loss, accuracy = model.evaluate(bundle.test_ds, verbose=0)

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in bundle.test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().astype(np.int32).tolist())

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(bundle.class_names)).numpy()
    per_class_accuracy: dict[str, float] = {}
    for idx, class_name in enumerate(bundle.class_names):
        class_total = cm[idx, :].sum()
        class_correct = cm[idx, idx]
        per_class_accuracy[class_name] = float(class_correct / class_total) if class_total > 0 else 0.0

    # Compute precision, recall, and F1 per class and macro/micro averages
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision_per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(tp),
        where=(precision_per_class + recall_per_class) > 0,
    )

    per_class_precision = {name: float(p) for name, p in zip(bundle.class_names, precision_per_class.tolist())}
    per_class_recall = {name: float(r) for name, r in zip(bundle.class_names, recall_per_class.tolist())}
    per_class_f1 = {name: float(f) for name, f in zip(bundle.class_names, f1_per_class.tolist())}

    # Macro averages: mean of per-class metrics
    precision_macro = float(np.mean(precision_per_class)) if precision_per_class.size > 0 else 0.0
    recall_macro = float(np.mean(recall_per_class)) if recall_per_class.size > 0 else 0.0
    f1_macro = float(np.mean(f1_per_class)) if f1_per_class.size > 0 else 0.0

    # Micro averages: compute from sums of TP/FP/FN
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    precision_micro = float(micro_tp / (micro_tp + micro_fp)) if (micro_tp + micro_fp) > 0 else 0.0
    recall_micro = float(micro_tp / (micro_tp + micro_fn)) if (micro_tp + micro_fn) > 0 else 0.0
    f1_micro = float(2 * precision_micro * recall_micro / (precision_micro + recall_micro)) if (precision_micro + recall_micro) > 0 else 0.0

    payload = {
        "run_dir": str(run_path),
        "metrics": {
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "test_precision_macro": precision_macro,
            "test_recall_macro": recall_macro,
            "test_f1_macro": f1_macro,
            "test_precision_micro": precision_micro,
            "test_recall_micro": recall_micro,
            "test_f1_micro": f1_micro,
        },
        "class_names": bundle.class_names,
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_accuracy,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
    }

    dump_json(payload, run_path / "evaluation.json")
    return payload
