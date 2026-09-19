"""
Shared logic for the Door subsystem: segmenting a continuous sensor stream
into door-open/close cycles and classifying each as Normal or Abnormal
resistance. Used by both the web app route and the standalone predict.py
script, so both ever behave identically.
"""
import os

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "door_model.joblib")
_bundle = None


def _get_model():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def parse_dt(s):
    y, mo, d, h, mi, se, ms = map(int, str(s).split("-"))
    return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=se, microsecond=ms * 1000)


def split_into_segments(df, gap_thresh_sec=1.0):
    diffs = df["ts"].diff()
    jump_positions = df.index[diffs.dt.total_seconds() > gap_thresh_sec]
    boundaries = [0] + list(jump_positions) + [len(df)]
    return [df.iloc[boundaries[i]:boundaries[i + 1]].reset_index(drop=True) for i in range(len(boundaries) - 1)]


def extract_features(chunk):
    cur = chunk["Motor current(mA)"].values.astype(float)
    pos = chunk["Door leaf position"].values.astype(float)
    volt = chunk["Motor Voltage(10mV)"].values.astype(float)
    bemf = chunk["Motor electrodynamic force"].values.astype(float)
    n = len(cur)
    d_cur = np.diff(cur)
    dd_cur = np.diff(d_cur)
    peaks, _ = find_peaks(cur, prominence=150)
    troughs, _ = find_peaks(-cur, prominence=150)
    return {
        "n_rows": n, "duration_s": n * 0.02,
        "cur_max": cur.max(), "cur_mean": cur.mean(), "cur_std": cur.std(), "cur_median": np.median(cur),
        "cur_total_variation": np.sum(np.abs(d_cur)) / n,
        "cur_jerk_std": dd_cur.std() if len(dd_cur) else 0,
        "n_peaks": len(peaks), "n_troughs": len(troughs),
        "n_extrema_per_row": (len(peaks) + len(troughs)) / n,
        "volt_std": volt.std(), "bemf_std": bemf.std(), "bemf_mean": bemf.mean(),
        "pos_range": pos.max() - pos.min(),
        "cur_skew": pd.Series(cur).skew(), "cur_kurt": pd.Series(cur).kurt(),
        "mid_cur_std": cur[n // 4: 3 * n // 4].std() if n >= 8 else cur.std(),
        "mid_extrema": (len(peaks[(peaks > n // 4) & (peaks < 3 * n // 4)])
                         + len(troughs[(troughs > n // 4) & (troughs < 3 * n // 4)])),
    }


REQUIRED_COLUMNS = [
    "Datetime", "Motor current(mA)", "Motor Voltage(10mV)", "Motor electrodynamic force",
    "Door leaf position",
]


def predict_from_csv(file_path_or_buffer):
    df = pd.read_csv(file_path_or_buffer)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s): {', '.join(missing)}")

    df["ts"] = df["Datetime"].apply(parse_dt)
    segments = split_into_segments(df)
    if not segments:
        raise ValueError("No door cycles could be found in this file.")

    bundle = _get_model()
    clf, feature_cols = bundle["model"], bundle["feature_cols"]

    rows = []
    for seg in segments:
        f = extract_features(seg)
        f["start_time"] = seg["Datetime"].iloc[0]
        f["end_time"] = seg["Datetime"].iloc[-1]
        rows.append(f)
    feat_df = pd.DataFrame(rows)

    X = feat_df[feature_cols]
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "start_time": feat_df["start_time"],
        "end_time": feat_df["end_time"],
        "prediction": np.where(preds == 1, "Abnormal resistance", "Normal"),
        "confidence": np.round(np.where(preds == 1, probs, 1 - probs), 3),
    })