"""Feature extraction for NebulaX PS3 Rail Corrugation (one 1-second CSV -> one feature row).

Layout of each CSV: col 0 = rotating speed (0/1 tooth toggle), then for every axle box a
Vibration and a Shock column (interleaved). 8 cars x 8 positions = 64 boxes.
Positions 1,3,5,7 = Side I ; positions 2,4,6,8 = Side II.
Only the VIBRATION channels are used (the shock channel did not help in CV).
"""
import argparse, os, re, sys, time
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis

FS = 10_000
TEETH = 90
WHEEL_DIAM_M = 0.85
NPERSEG = 1024
BAND_EDGES_HZ = [5, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000]
WL_EDGES_M = [0.02, 0.03, 0.05, 0.08, 0.125, 0.2, 0.4]   # corrugation wavelength bands (m)
EPS = 1e-12
VIB_RE = re.compile(r"Vibration of bearing in position (\d+) of car (\d+)", re.I)


def split_columns(df):
    speed_col = [c for c in df.columns if c.lower().startswith("rotating speed")][0]
    boxes = []
    for c in df.columns:
        m = VIB_RE.match(c)
        if m:
            boxes.append((int(m.group(2)), int(m.group(1)), c))
    boxes.sort()
    assert len(boxes) == 64, f"expected 64 vibration columns, got {len(boxes)}"
    cols = [b[2] for b in boxes]
    side = np.array([0 if b[1] % 2 == 1 else 1 for b in boxes])  # 0 = Side I, 1 = Side II
    return df[speed_col].to_numpy(float), df[cols].to_numpy(float), side


def speed_from_toggle(x):
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return 0.0, 0
    b = x > (lo + hi) / 2
    rising = int(np.sum(b[1:] & ~b[:-1]))
    v = (rising / (len(x) / FS)) / TEETH * np.pi * WHEEL_DIAM_M      # m/s
    return v, rising


def _cum_interp(f, C, x):
    return np.array([np.interp(x, f, C[:, j]) for j in range(C.shape[1])])


def per_box_features(V, v_ms):
    F = {}
    V = V - V.mean(axis=0, keepdims=True)
    std = V.std(axis=0)
    amax = np.abs(V).max(axis=0)
    q = np.percentile(V, [5, 25, 75, 95], axis=0)
    F["log_std"] = np.log10(std + EPS)
    F["log_absmax"] = np.log10(amax + EPS)
    F["log_p2p"] = np.log10(V.max(0) - V.min(0) + EPS)
    F["log_iqr"] = np.log10(q[2] - q[1] + EPS)
    F["log_r5_95"] = np.log10(q[3] - q[0] + EPS)
    F["skew"] = skew(V, axis=0)
    F["kurt"] = kurtosis(V, axis=0)
    F["crest"] = amax / (std + EPS)

    f, P = welch(V, fs=FS, nperseg=NPERSEG, axis=0)
    df = f[1] - f[0]
    tm = (f >= BAND_EDGES_HZ[0]) & (f < BAND_EDGES_HZ[-1])
    Ptot = P[tm].sum(0) * df + EPS
    F["log_bp_total"] = np.log10(Ptot)
    for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        m = (f >= lo) & (f < hi)
        bp = P[m].sum(0) * df
        F[f"log_bp_{lo}_{hi}"] = np.log10(bp + EPS)
        F[f"rel_bp_{lo}_{hi}"] = bp / Ptot
    Pm, fm = P[tm], f[tm]
    S = Pm.sum(0) + EPS
    cen = (fm[:, None] * Pm).sum(0) / S
    F["spec_centroid"] = cen
    F["spec_bandwidth"] = np.sqrt((((fm[:, None] - cen) ** 2) * Pm).sum(0) / S)
    p = Pm / S
    F["spec_entropy"] = -(p * np.log(p + EPS)).sum(0) / np.log(len(fm))
    F["spec_flatness"] = np.exp(np.log(Pm + EPS).mean(0)) / (Pm.mean(0) + EPS)
    F["spec_rolloff85"] = fm[(np.cumsum(Pm, 0) / S >= 0.85).argmax(0)]
    F["dom_freq"] = fm[Pm.argmax(0)]
    F["dom_amp"] = np.log10(Pm.max(0) + EPS)
    cm = (f >= 200) & (f <= 3000)
    Pc, fc = P[cm], f[cm]
    dcf = fc[Pc.argmax(0)]
    F["dom_freq_200_3000"] = dcf
    F["dom_amp_200_3000"] = np.log10(Pc.max(0) + EPS)
    F["peak_ratio_200_3000"] = np.log10(Pc.max(0) / (np.median(Pc, 0) + EPS) + EPS)

    # wavelength-domain (f = v / lambda): corrugation has a fixed wavelength, so the
    # excited frequency scales with train speed.
    nb = len(WL_EDGES_M) - 1
    if v_ms > 0:
        C = np.cumsum(P, axis=0) * df
        for k, (a, b) in enumerate(zip(WL_EDGES_M[:-1], WL_EDGES_M[1:])):
            flo, fhi = max(v_ms / b, 5.0), min(v_ms / a, 5000.0)
            e = 0.0 * Ptot if fhi <= flo else (_cum_interp(f, C, fhi) - _cum_interp(f, C, flo))
            F[f"wl_rel_{k}"] = e / Ptot
        F["dom_wavelength"] = np.log10(v_ms / np.maximum(dcf, 1.0))
    else:
        for k in range(nb):
            F[f"wl_rel_{k}"] = np.full(V.shape[1], np.nan)
        F["dom_wavelength"] = np.full(V.shape[1], np.nan)
    return F


def extract_file(path):
    df = pd.read_csv(path)
    meta = {"meta_rows": len(df), "meta_cols": df.shape[1],
            "meta_nan": int(df.isna().sum().sum()),
            "meta_inf": int(np.isinf(df.to_numpy(float)).sum())}
    spd, V, side = split_columns(df)
    v, rising = speed_from_toggle(spd)
    F = per_box_features(V, v)
    row = {"speed_kmh": v * 3.6, "speed_toggles": rising, "log_speed": np.log10(v * 3.6 + 1.0)}
    I, II = side == 0, side == 1
    for name, a in F.items():
        mi, mii, xi, xii = a[I].mean(), a[II].mean(), a[I].max(), a[II].max()
        row[f"{name}__I_mean"], row[f"{name}__II_mean"] = mi, mii
        row[f"{name}__I_max"], row[f"{name}__II_max"] = xi, xii
        row[f"{name}__diff_mean"], row[f"{name}__diff_max"] = mi - mii, xi - xii
    return row, meta


def extract_dir(files, verbose=True):
    rows, metas = [], []
    t0 = time.time()
    for i, p in enumerate(files):
        r, m = extract_file(p)
        fid = os.path.basename(p)
        rows.append({"file_id": fid, **r}); metas.append({"file_id": fid, **m})
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(files)} files  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(metas)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def list_csvs(path):
    if os.path.isdir(path):
        fs = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".csv")]
    else:
        fs = [path]
    return sorted(fs, key=lambda p: natural_key(os.path.basename(p)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory of CSVs")
    ap.add_argument("--output", required=True, help="features CSV to write")
    ap.add_argument("--meta-output", default=None)
    a = ap.parse_args()
    feats, meta = extract_dir(list_csvs(a.input))
    feats.to_csv(a.output, index=False)
    if a.meta_output:
        meta.to_csv(a.meta_output, index=False)
    print("wrote", a.output, feats.shape)
