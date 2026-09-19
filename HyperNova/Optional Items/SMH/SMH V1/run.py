#!/usr/bin/env python3
"""
run_all.py
Pipeline: install -> (maybe delete config) -> train
          -> CV report -> (maybe delete output) -> predict

Test/ is NEVER read by this script.

Run from the project root:
    python run_all.py
"""

import os
import subprocess
import sys

# ---------- Config ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(ROOT, "code")
OUTPUT_CSV = os.path.join(ROOT, "shm_predictions.csv")
CONFIG_FILE = os.path.join(ROOT, "model", "config.json")
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")

# Which set to predict on
PREDICT_INPUT = "Test"
PREDICT_INPUT_DIR = os.path.join(ROOT, PREDICT_INPUT)


def run(cmd, cwd=None):
    """Run a command, stream output, return exit code."""
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def ask_yes_no(prompt):
    """Ask a Y/N question. Returns True for yes."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", ""):
            return False
        print("Please answer Y or N.")


def main():
    # ---------- Step 1: pip install ----------
    print("=" * 60)
    print(" Step 1: Install Python dependencies")
    print("=" * 60)
    rc = run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS])
    if rc != 0:
        print("ERROR: pip install failed.")
        sys.exit(1)

    # ---------- Step 2: maybe delete config, then train ----------
    print()
    print("=" * 60)
    print(" Step 2: Train the model")
    print("=" * 60)

    if os.path.exists(CONFIG_FILE):
        print(f' The trained model config "{CONFIG_FILE}" already exists.')
        if ask_yes_no("Do you want to DELETE it and retrain from scratch? [Y/N]: "):
            print(f'Deleting "{CONFIG_FILE}" ...')
            os.remove(CONFIG_FILE)
        else:
            print(f'Keeping existing "{CONFIG_FILE}". Training will be skipped.')

    rc = run([sys.executable, "train_model.py"], cwd=CODE_DIR)
    if rc != 0:
        print("ERROR: Training failed.")
        sys.exit(1)

    # ---------- Step 3: CV report ----------
    print()
    print("=" * 60)
    print(" Step 3: Cross-validation report (Train only)")
    print("=" * 60)
    rc = run([sys.executable, "cv_report.py"], cwd=CODE_DIR)
    if rc != 0:
        print("ERROR: CV report failed.")
        sys.exit(1)

    # ---------- Step 4: predictions ----------
    print()
    print("=" * 60)
    print(" Step 4: Generate predictions")
    print("=" * 60)

    if os.path.exists(OUTPUT_CSV):
        print(f' The output file "{OUTPUT_CSV}" already exists.')
        if not ask_yes_no("Do you want to DELETE it and regenerate? [Y/N]: "):
            print(f'Keeping existing "{OUTPUT_CSV}". Skipping prediction step.')
            sys.exit(0)
        print(f'Deleting "{OUTPUT_CSV}" ...')
        os.remove(OUTPUT_CSV)

    rc = run(
        [
            sys.executable,
            "predict.py",
            "--input", os.path.join("..", PREDICT_INPUT),
            "--output", os.path.join("..", os.path.basename(OUTPUT_CSV)),
        ],
        cwd=CODE_DIR,
    )
    if rc != 0:
        print("ERROR: Prediction failed.")
        sys.exit(1)

    # ---------- Done ----------
    print()
    print("=" * 60)
    print(" DONE!")
    print(f" Predictions saved to: {OUTPUT_CSV}")
    print(f" Test\\ folder was NOT read (input was {PREDICT_INPUT}).")
    print("=" * 60)


if __name__ == "__main__":
    main()