"""
Wraps ACV's own predict.py (unmodified, run as a subprocess) so the web app
calls exactly the same code path that was independently verified against
the labeled training data.
"""
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICT_SCRIPT = os.path.join(HERE, "acv_core.py")


def predict_acv(uploaded_file, original_filename):
    """uploaded_file: a Flask FileStorage. Returns (file_id, ranked_cars_list)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, original_filename)
        uploaded_file.save(in_path)
        out_path = os.path.join(tmpdir, "out.csv")

        result = subprocess.run(
            [sys.executable, PREDICT_SCRIPT, "--input", in_path, "--output", out_path],
            capture_output=True, text=True, cwd=HERE,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ACV prediction failed: {result.stderr.strip()[-500:]}")

        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if not rows:
            raise RuntimeError("ACV prediction produced no output.")
        row = rows[0]
        ranked = row["ranked_cars"].split("|")
        return row["file_id"], ranked
