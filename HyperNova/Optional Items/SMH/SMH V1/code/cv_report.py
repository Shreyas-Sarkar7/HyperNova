import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from features import load_signal, get_cycles

TRAIN_DIR = "../Train"
LABELS_FILE = "../Train_Labels.csv"
CONFIG = "../model/config.json"

print("[1/4] Loading config and labels ...")
with open(CONFIG) as f:
    cfg = json.load(f)
m, alpha, beta = cfg["m"], cfg["alpha"], cfg["beta"]
print(f"  Loaded config: m={m}, alpha={alpha:.3e}, beta={beta:.4f}")

labels = pd.read_csv(LABELS_FILE)
filenames = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith(".csv")])
labels = labels.set_index("filename").loc[filenames]
y_true = labels["damage"].values
print(f"  {len(filenames)} training files loaded")

print("\n[2/4] Computing Miner proxies for all training files ...")
proxies = []
for fname in tqdm(filenames, desc="  Proxies"):
    sig = load_signal(os.path.join(TRAIN_DIR, fname))
    amps, counts = get_cycles(sig)
    p = float(np.sum(counts * (amps ** m))) if len(amps) else 0.0
    proxies.append(p)
proxies = np.array(proxies)

print("\n[3/4] Computing in-sample and CV scores ...")
y_pred = alpha * (proxies ** beta)
mape_in = mean_absolute_percentage_error(y_true, y_pred)
print(f"  In-sample MAPE: {mape_in:.4f}  score: {max(0, 1-mape_in):.4f}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(kf.split(proxies))
mapes = []
for _, val_idx in tqdm(fold_indices, desc="  CV folds"):
    y_pred_val = alpha * (proxies[val_idx] ** beta)
    mapes.append(mean_absolute_percentage_error(y_true[val_idx], y_pred_val))
cv_mape = float(np.mean(mapes))
print(f"  Fixed-parameter CV MAPE: {cv_mape:.4f}  score: {max(0, 1-cv_mape):.4f}")

print("\n[4/4] Per-file breakdown:")
for fname, yt, yp in zip(filenames, y_true, y_pred):
    err = abs(yt - yp) / abs(yt)
    print(f"  {fname:15s} true={yt:.3e}  pred={yp:.3e}  err={err*100:6.1f}%")