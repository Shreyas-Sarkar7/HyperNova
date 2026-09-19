"""
Wraps SHM's own predict.py (unmodified, run as a subprocess with the cwd
set to code/, since it locates its model config via a relative path).
Accepts multiple uploaded CSVs (one per vibration/stress reading file).
"""
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(HERE, "code")
PREDICT_SCRIPT = os.path.join(CODE_DIR, "predict.py")


def predict_shm(uploaded_files):
    """uploaded_files: list of Flask FileStorage objects (each a .csv).
    Returns a list of {file_id, prediction} dicts (prediction = cumulative
    fatigue damage proxy, roughly 0-1+)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in uploaded_files:
            f.save(os.path.join(tmpdir, f.filename))
        out_path = os.path.join(tmpdir, "out.csv")

        result = subprocess.run(
            [sys.executable, PREDICT_SCRIPT, "--input", tmpdir, "--output", out_path],
            capture_output=True, text=True, cwd=CODE_DIR,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SHM prediction failed: {result.stderr.strip()[-500:]}")

        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        return rows
