import numpy as np
import pandas as pd
import rainflow

def load_signal(filepath):
    """Load a CSV and return the single stress column as a 1D numpy array."""
    df = pd.read_csv(filepath)
    if df.shape[1] > 1:
        stress_col = df.select_dtypes(include=[np.number]).columns[0]
        return df[stress_col].values
    return df.iloc[:, 0].values

def get_cycles(signal):
    """Rainflow count. Returns (amplitudes, counts).
    Handles both (range, count) and (range, mean, count) formats."""
    cycles = rainflow.count_cycles(signal)
    amps = []
    counts = []
    for c in cycles:
        if len(c) == 2:
            rng, cnt = c
        elif len(c) == 3:
            rng, _, cnt = c
        else:
            raise ValueError(f"Unexpected rainflow tuple length: {len(c)}")
        amps.append(rng / 2.0)
        counts.append(cnt)
    return np.array(amps), np.array(counts)

def miner_proxy(signal, m):
    """Sum(n_i * amp_i^m) for a given exponent m."""
    amps, counts = get_cycles(signal)
    if len(amps) == 0:
        return 0.0
    return float(np.sum(counts * (amps ** m)))

def extract_features(signal):
    """Return a dict of features for one stress signal."""
    feats = {}
    feats["mean"] = np.mean(signal)
    feats["std"] = np.std(signal)
    feats["min"] = np.min(signal)
    feats["max"] = np.max(signal)
    feats["range"] = feats["max"] - feats["min"]
    feats["rms"] = np.sqrt(np.mean(signal ** 2))
    feats["skew"] = float(pd.Series(signal).skew())
    feats["kurtosis"] = float(pd.Series(signal).kurtosis())

    amps, counts = get_cycles(signal)
    feats["n_cycles"] = counts.sum() if len(counts) else 0.0
    feats["max_amp"] = amps.max() if len(amps) else 0.0
    feats["sum_amp"] = np.sum(counts * amps) if len(amps) else 0.0
    for p in [2, 3, 4, 5, 6]:
        feats[f"sum_amp_p{p}"] = np.sum(counts * (amps ** p)) if len(amps) else 0.0
    for m in [3, 4, 5, 6, 7, 8, 9, 10]:
        feats[f"proxy_m{m}"] = miner_proxy(signal, m)
    return feats