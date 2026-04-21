# MobileNetV2 Satellite Image Classification

## Objective

Develop a reproducible, script-based project that uses **transfer learning with MobileNetV2** to classify satellite imagery.

## What Must Be Included

1. Use transfer learning with MobileNetV2.
2. Compare performance across multiple image sizes.
	 - Note: resized image datasets are not in the repo yet, but will follow the same format as 28x28
3. Compare augmentation vs. no augmentation.
4. Use a **Python scripts approach**, not notebook-first development.

## Development Approach (Important)
- Use the transfer learning example from transfer_learning.ipynb
- Prioritize modular Python scripts for:
	- data loading and preprocessing
	- model creation
	- training
	- evaluation
	- experiment orchestration
- Keep code reproducible and easy to rerun with different settings.
- Avoid hardcoded paths and magic numbers.

## Expected Project Outcomes

- A training pipeline that can run multiple experiment configurations.
- Clear comparison results for:
	- image size variants
	- augmentation vs. no augmentation
- Saved model artifacts/checkpoints so experiments do not need to be retrained unnecessarily.

## Experiment Matrix

At minimum, support experiment combinations like:

- Image size A + augmentation
- Image size A + no augmentation
- Image size B + augmentation
- Image size B + no augmentation

Add more sizes if feasible.

## Reproducibility and Artifact Requirements

- Save best model weights for each experiment setting.
- Save full training checkpoints (model + optimizer + scheduler + epoch/step when possible).
- Save metrics and config for each run so results are traceable.

## Deliverables to Produce

Please produce:

1. Suggested folder structure for script-based workflow.
2. Core Python files and their responsibilities.
3. CLI-style commands to run training/evaluation experiments.
4. Logging and checkpoint strategy.
5. Evaluation/reporting workflow for comparing experiments.
6. Minimal README usage section.

## Constraints and Preferences

- Keep implementation clean and easy to explain in a class setting.
- Prefer readability over unnecessary complexity.
- Use common, well-supported libraries.