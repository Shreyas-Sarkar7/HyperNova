import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm

from features import load_signal, get_cycles

TRAIN_DIR = "../Train"
LABELS_FILE = "../Train_Labels.csv"
MODEL_DIR = "../model"
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
FORCE_RETRAIN = "--force" in sys.argv

if os.path.exists(CONFIG_PATH) and not FORCE_RETRAIN:
    print(f"[SKIP] Model already exists at {CONFIG_PATH}")
    print("       Delete it or run 'python train_model.py --force' to retrain.")
    sys.exit(0)

t0 = time.time()

print("[1/4] Loading training data ...")
labels = pd.read_csv(LABELS_FILE)
filenames = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith(".csv")])
labels = labels.set_index("filename").loc[filenames]
y_true = labels["damage"].values

signals = []
for f in tqdm(filenames, desc="  Reading CSVs"):
    signals.append(load_signal(os.path.join(TRAIN_DIR, f)))
print(f"  Loaded {len(signals)} files, damage range [{y_true.min():.3e}, {y_true.max():.3e}]")

print("\n[2/4] Precomputing rainflow cycles (once per file) ...")
cached_cycles = []
for sig in tqdm(signals, desc="  Rainflow"):
    cached_cycles.append(get_cycles(sig))

m_grid = np.linspace(2.0, 15.0, 40)
print(f"  Building proxy matrix for {len(m_grid)} m-values ...")
proxy_matrix = np.zeros((len(cached_cycles), len(m_grid)))
for i, (amps, counts) in enumerate(tqdm(cached_cycles, desc="  Proxies")):
    if len(amps) == 0:
        continue
    for j, m in enumerate(m_grid):
        proxy_matrix[i, j] = np.sum(counts * (amps ** m))

def fit_power_law_from_proxies(proxies, y):
    log_proxy = np.log(proxies + 1e-12).reshape(-1, 1)
    log_y = np.log(y + 1e-12)
    lr = LinearRegression()
    lr.fit(log_proxy, log_y)
    alpha = np.exp(lr.intercept_)
    beta = lr.coef_[0]
    y_pred = alpha * (proxies ** beta)
    mape = mean_absolute_percentage_error(y, y_pred)
    return alpha, beta, mape

print("\n[3/4] Cross-validating power-law model over m values ...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(kf.split(signals))

best_m = None
best_score = -1
for j, m in enumerate(tqdm(m_grid, desc="  m grid search")):
    proxies = proxy_matrix[:, j]
    mapes = []
    for train_idx, val_idx in fold_indices:
        alpha_f, beta_f, _ = fit_power_law_from_proxies(proxies[train_idx], y_true[train_idx])
        y_pred = alpha_f * (proxies[val_idx] ** beta_f)
        mapes.append(mean_absolute_percentage_error(y_true[val_idx], y_pred))
    cv_mape = float(np.mean(mapes))
    score = max(0, 1 - cv_mape)
    if score > best_score:
        best_score = score
        best_m = m

print(f"\n  Best power-law: m={best_m:.3f}, CV MAPE={1-best_score:.4f}, score={best_score:.4f}")

j_best = int(np.argmin(np.abs(m_grid - best_m)))
alpha, beta, _ = fit_power_law_from_proxies(proxy_matrix[:, j_best], y_true)

config = {
    "model_type": "power_law",
    "m": float(best_m),
    "alpha": float(alpha),
    "beta": float(beta)
}

print("\n[4/4] Saving model ...")
os.makedirs(MODEL_DIR, exist_ok=True)
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print(f"  Saved config to {CONFIG_PATH}")
print(f"  Final CV score: {best_score:.4f}")
print(f"  Total time: {time.time() - t0:.1f} s")