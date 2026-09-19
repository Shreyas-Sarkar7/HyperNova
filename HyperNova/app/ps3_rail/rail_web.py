"""
Wraps Rail Corrugation's own predict.py (unmodified, run as a subprocess),
accepting multiple uploaded CSVs (one per rail-corrugation reading file).
"""
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICT_SCRIPT = os.path.join(HERE, "predict.py")
MODEL_PATH = os.path.join(HERE, "model.joblib")


def predict_rail(uploaded_files):
    """uploaded_files: list of Flask FileStorage objects (each a .csv).
    Returns a list of {file_id, prediction} dicts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in uploaded_files:
            f.save(os.path.join(tmpdir, f.filename))
        out_path = os.path.join(tmpdir, "out.csv")

        result = subprocess.run(
            [sys.executable, PREDICT_SCRIPT, "--input", tmpdir, "--output", out_path,
             "--model", MODEL_PATH],
            capture_output=True, text=True, cwd=HERE,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Rail Corrugation prediction failed: {result.stderr.strip()[-500:]}")

        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        return rows
