from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class TrainConfig:
    dataset_dir: str
    image_size: int
    output_dir: str = "outputs"
    run_name: str = ""
    batch_size: int = 32
    epochs: int = 20
    seed: int = 42
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    augmentation: bool = False
    learning_rate: float = 1e-3
    dropout_rate: float = 0.2
    base_trainable: bool = False
    early_stopping_patience: int = 6

    def validate(self) -> None:
        split_sum = self.train_split + self.val_split + self.test_split
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {split_sum:.6f}")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatrixConfig:
    data_root: str
    output_dir: str = "outputs"
    image_size_dirs: list[str] | None = None
    augment_options: list[bool] | None = None
    batch_size: int = 32
    epochs: int = 20
    seed: int = 42
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    learning_rate: float = 1e-3
    dropout_rate: float = 0.2
    base_trainable: bool = False
    early_stopping_patience: int = 6

    def __post_init__(self) -> None:
        if self.image_size_dirs is None:
            self.image_size_dirs = []
        if self.augment_options is None:
            self.augment_options = [False, True]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dump_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
