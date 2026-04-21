from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random

import numpy as np
from PIL import Image
import tensorflow as tf


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class DatasetBundle:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    class_names: list[str]
    class_to_index: dict[str, int]
    split_counts: dict[str, int]


def _class_sort_key(name: str) -> tuple[int, str]:
    return (0, f"{int(name):09d}") if name.isdigit() else (1, name)


def _list_image_paths(class_dir: Path) -> list[Path]:
    all_paths = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(all_paths)


def discover_dataset(dataset_dir: str | Path) -> tuple[list[str], list[Path], list[int]]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    class_dirs = [p for p in dataset_path.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {dataset_path}")

    class_names = sorted((p.name for p in class_dirs), key=_class_sort_key)
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    image_paths: list[Path] = []
    labels: list[int] = []
    for class_name in class_names:
        class_path = dataset_path / class_name
        class_files = _list_image_paths(class_path)
        if not class_files:
            continue
        image_paths.extend(class_files)
        labels.extend([class_to_index[class_name]] * len(class_files))

    if not image_paths:
        raise ValueError(f"No supported image files found in {dataset_path}")

    return class_names, image_paths, labels


def split_paths(
    image_paths: list[Path],
    labels: list[int],
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> dict[str, tuple[list[str], list[int]]]:
    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels lengths do not match")

    indices = list(range(len(image_paths)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    num_samples = len(indices)
    train_end = int(num_samples * train_split)
    val_end = train_end + int(num_samples * val_split)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def pick(idxs: Iterable[int]) -> tuple[list[str], list[int]]:
        p = [str(image_paths[i]) for i in idxs]
        y = [labels[i] for i in idxs]
        return p, y

    return {
        "train": pick(train_idx),
        "val": pick(val_idx),
        "test": pick(test_idx),
    }


def _load_image_py(path: bytes, image_size: int) -> np.ndarray:
    image_path = path.decode("utf-8")
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
    return arr


def _build_dataset(paths: list[str], labels: list[int], image_size: int, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    path_tensor = tf.constant(paths, dtype=tf.string)
    label_tensor = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((path_tensor, label_tensor))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    def load_map(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.numpy_function(func=lambda p: _load_image_py(p, image_size), inp=[path], Tout=tf.float32)
        image.set_shape((image_size, image_size, 3))
        return image, label

    ds = ds.map(load_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def create_datasets(
    dataset_dir: str,
    image_size: int,
    batch_size: int,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> DatasetBundle:
    class_names, image_paths, labels = discover_dataset(dataset_dir)
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    splits = split_paths(
        image_paths=image_paths,
        labels=labels,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    if not train_paths or not val_paths or not test_paths:
        raise ValueError(
            "One or more splits are empty. Adjust split ratios or provide more data. "
            f"Counts: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}"
        )

    train_ds = _build_dataset(train_paths, train_labels, image_size, batch_size, shuffle=True, seed=seed)
    val_ds = _build_dataset(val_paths, val_labels, image_size, batch_size, shuffle=False, seed=seed)
    test_ds = _build_dataset(test_paths, test_labels, image_size, batch_size, shuffle=False, seed=seed)

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        class_names=class_names,
        class_to_index=class_to_index,
        split_counts={"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
    )
