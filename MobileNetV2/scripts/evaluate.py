from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mobilenet_pipeline.evaluate import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a previously trained run.")
    parser.add_argument("--run-dir", required=True, help="Path to a run output directory.")
    args = parser.parse_args()

    result = evaluate_run(args.run_dir)
    print(f"Evaluation complete for: {result['run_dir']}")
    print(f"Test accuracy: {result['metrics']['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
