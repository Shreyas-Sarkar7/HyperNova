#!/usr/bin/env python3
"""Train the ACV leak-localisation scorer.

python train.py --data_dir <folder with acv_case_*.xlsx> --labels Train_Labels.csv [--out acv_model.json]
Defaults: data_dir="data", labels="<data_dir>/Train_Labels.csv", out="acv_model.json" next to this file.

Reads every file listed in the labels CSV that exists in --data_dir (missing files are
skipped with a warning). Fits non-negative feature weights with a listwise (softmax over
the 8 cars) logistic model + L2, reports leave-one-case-out validation with the hackathon
score (n-(r-1))/n, and saves the final model trained on all cases.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from predict import (DEFAULT_CFG, FEATURES, car_features, rank_cars, read_table,
                     residual_matrix, zscore_across_cars)


def load_labels(path):
    lab = pd.read_csv(path, dtype=str)                       # keep '01' as text
    lab.columns = [c.strip().lower() for c in lab.columns]
    return dict(zip(lab["filename"].str.strip(), lab["faulty_car"].str.strip()))


def same_car(a, b):
    try:
        return int(a) == int(b)
    except ValueError:
        return str(a) == str(b)


def build_dataset(data_dir, labels, cfg):
    """-> list of dict(name, cars, Z (n_cars x n_feat), y (index of faulty car))"""
    ds = []
    for fname, car in labels.items():
        p = os.path.join(data_dir, fname)
        if not os.path.exists(p):
            print(f"{fname} listed in labels but not found in {data_dir}, skipped")
            continue
        res, cars, info = residual_matrix(read_table(p), cfg)
        if res is None:
            print(f"{fname}: {info}, skipped")
            continue
        Z = zscore_across_cars(car_features(res, cfg))
        idx = [i for i, c in enumerate(Z.index) if same_car(c, car)]
        if not idx:
            print(f"{fname}: labelled car {car} not in headers {list(Z.index)}, skipped")
            continue
        ds.append({"name": fname, "cars": list(Z.index), "Z": Z.values.astype(float), "y": idx[0]})
        print(f"loaded {fname}: cars={len(cars)} rows={len(res)}  {info}")
    return ds


def fit_weights(ds, lam=0.3, iters=3000, lr=0.05):
    """Projected gradient descent on listwise NLL + lam*||w||^2, with w >= 0
    (hotter-than-sisters can only make a car more suspicious)."""
    k = len(FEATURES)
    w = np.full(k, 0.5)
    for _ in range(iters):
        g = 2 * lam * w
        for d in ds:
            s = d["Z"] @ w
            p = np.exp(s - s.max()); p /= p.sum()
            onehot = np.zeros_like(p); onehot[d["y"]] = 1.0
            g = g + d["Z"].T @ (p - onehot) / len(ds)
        w = np.maximum(w - lr * g, 0.0)
    return w


def file_score(rank, n):
    return (n - (rank - 1)) / n


def rank_of_true(scores, cars, y):
    order = rank_cars(pd.Series(scores, index=cars))
    return order.index(cars[y]) + 1


def evaluate(ds, weight_fn, label):
    ranks, scs = [], []
    for i, d in enumerate(ds):
        train = [x for j, x in enumerate(ds) if j != i]
        w = weight_fn(train)
        r = rank_of_true(d["Z"] @ w, d["cars"], d["y"])
        ranks.append(r); scs.append(file_score(r, len(d["cars"])))
    print(f"  {label:<34} ranks={ranks}  mean score={np.mean(scs):.3f}")
    return np.mean(scs)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=os.path.join(here, "acv_model.json"))
    ap.add_argument("--lam", type=float, default=0.3)
    a = ap.parse_args()
    if a.labels is None:
        a.labels = os.path.join(a.data_dir, "Train_Labels.csv")
    if not os.path.isdir(a.data_dir):
        sys.exit(f"folder '{a.data_dir}' not found. Create a folder called 'data' next to train.py "
                 f"and put the training .xlsx files + Train_Labels.csv inside.")
    if not os.path.exists(a.labels):
        sys.exit(f"labels file '{a.labels}' not found.")

    cfg = dict(DEFAULT_CFG)
    ds = build_dataset(a.data_dir, load_labels(a.labels), cfg)
    if len(ds) < 2:
        sys.exit("need at least 2 labelled cases to train")
    unit = lambda idx: (lambda tr: np.eye(len(FEATURES))[idx])
    print(f"\nleave-one-case-out validation on {len(ds)} cases (rank of true faulty car):")
    for i, f in enumerate(FEATURES):
        evaluate(ds, unit(i), f"single feature: {f}")
    evaluate(ds, lambda tr: np.ones(len(FEATURES)), "equal weights (all features)")
    evaluate(ds, lambda tr: fit_weights(tr, a.lam), f"learned weights (lam={a.lam})")

    w = fit_weights(ds, a.lam)
    weights = {f: round(float(v), 4) for f, v in zip(FEATURES, w)}
    print("\nfinal weights (trained on all cases):", weights)
    print("training-set ranks:", [rank_of_true(d["Z"] @ w, d["cars"], d["y"]) for d in ds])
    with open(a.out, "w") as fh:
        json.dump({"weights": weights, "config": cfg, "features": FEATURES,
                   "trained_on": [d["name"] for d in ds]}, fh, indent=2)
    print(f"\nmodel saved to {a.out}")


if __name__ == "__main__":
    main()
