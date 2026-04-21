# MobileNetV2 Transfer Learning Pipeline

Script-first, reproducible transfer learning workflow for satellite image classification with MobileNetV2.

## Suggested Folder Structure

```text
MobileNetV2/
  28x28/
    <class_id>/
      *.tif
  56x56/                    # optional future dataset size
  scripts/
    train.py
    evaluate.py
    run_matrix.py
  src/
    mobilenet_pipeline/
      __init__.py
      config.py
      data.py
      model.py
      train.py
      evaluate.py
      experiments.py
  outputs/
    <run_name>/
      config.json
      metrics.csv
      run_summary.json
      best.weights.h5
      final_model.keras
      checkpoints/
        epoch_001.keras
        ...
    matrix_results.json
  transfer_learning.ipynb
  requirements.txt
  README.md
```

## Core Files and Responsibilities

- `src/mobilenet_pipeline/data.py`: dataset discovery, deterministic split, and tf.data loading.
- `src/mobilenet_pipeline/model.py`: MobileNetV2 transfer model definition and compile config.
- `src/mobilenet_pipeline/train.py`: training loop, callbacks, checkpoints, metrics, and saved config.
- `src/mobilenet_pipeline/evaluate.py`: post-training evaluation with confusion matrix and per-class accuracy.
- `src/mobilenet_pipeline/experiments.py`: matrix orchestration across size folders and augmentation settings.
- `scripts/train.py`: CLI for a single training run.
- `scripts/evaluate.py`: CLI for evaluating one saved run.
- `scripts/run_matrix.py`: CLI for experiment matrix execution.

## Install

From `MobileNetV2/`:

```bash
pip install -r requirements.txt
```

## CLI Commands

### Single run, no augmentation

```bash
python scripts/train.py \
  --dataset-dir ./28x28 \
  --image-size 28 \
  --run-name size_28x28_aug_false \
  --epochs 20
```

### Single run, with augmentation

```bash
python scripts/train.py \
  --dataset-dir ./28x28 \
  --image-size 28 \
  --augmentation \
  --run-name size_28x28_aug_true \
  --epochs 20
```

### Evaluate a saved run

```bash
python scripts/evaluate.py --run-dir ./outputs/size_28x28_aug_true
```

### Run experiment matrix (sizes x augmentation)

```bash
python scripts/run_matrix.py \
  --data-root . \
  --size-dirs 28x28 56x56 \
  --augment-options false true \
  --epochs 20
```

If only `28x28` exists now:

```bash
python scripts/run_matrix.py --data-root . --size-dirs 28x28 --augment-options false true --epochs 20
```

## Logging and Checkpoint Strategy

- `metrics.csv`: per-epoch train/validation metrics.
- `best.weights.h5`: best validation accuracy weights.
- `checkpoints/epoch_XXX.keras`: full model checkpoints each epoch.
- `final_model.keras`: final saved model artifact.
- `config.json`: full run configuration for reproducibility.
- `run_summary.json`: history, class mapping, split counts, and test metrics.
- `outputs/matrix_results.json`: aggregated results across experiment matrix runs.

## Evaluation and Reporting Workflow

1. Run matrix with `scripts/run_matrix.py`.
2. Open `outputs/matrix_results.json` and compare `test_accuracy` by condition.
3. For each key run, execute `scripts/evaluate.py` to generate `evaluation.json`.
4. Use confusion matrix and per-class accuracy to discuss class-level behavior.

## Notes

- Dataset folders are expected in class-subfolder format (`<size>/<class_id>/*.tif`).
- Image size is inferred from folder name in matrix mode (`28x28 -> 28`).
- Splits are deterministic with `--seed` for reproducibility.
