#!/usr/bin/env python3
"""ACV refrigerant-leak localisation inference.

python predict.py --input <file.xlsx | folder> --output acv_predictions.csv [--model acv_model.json]
Defaults: input="test", output="acv_predictions.csv", model="acv_model.json" next to this file.

A leaking car cools poorly, so its indoor temp sits above its cooling
set-point relative to its 7 sister cars while in cooling mode, and the gap
tends to grow with time. Features are computed relative to the train median
at each timestamp, z-scored across cars in the file, and combined with
weights from acv_model.json (see train.py).
"""
import argparse
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

CAR_RE = re.compile(r"^\s*Car\s+(\w+)\s*-\s*(.+?)\s*$", re.I)

DEFAULT_CFG = {
    "late_frac": 0.4,      # last 40% of a car's valid samples = "late window"
    "res_thresh": 0.5,     # degC above train median counted as "hot"
    "min_cars": 4,         # min cars with valid data at a timestamp to form a median
    "min_samples": 50,     # min valid samples for a car to get features
}
FEATURES = ["res_mean", "res_p90", "frac_hot", "late_mean"]
DEFAULT_WEIGHTS = {"res_mean": 1.0, "res_p90": 0.5, "frac_hot": 0.5, "late_mean": 1.0}


def read_table(path):
    if path.lower().endswith((".csv", ".txt")):
        return pd.read_csv(path)
    return pd.read_excel(path)


def parse_columns(df):
    colmap = {}
    for c in df.columns:
        m = CAR_RE.match(str(c))
        if m:
            colmap[(m.group(1), m.group(2).strip())] = c
    cars = sorted({k[0] for k in colmap}, key=lambda s: (len(s), s))
    return cars, colmap


def find_param(params, must, none_of=(), prefer=()):
    cands = [p for p in params
             if all(t in p.lower() for t in must) and not any(t in p.lower() for t in none_of)]
    for pref in prefer:
        for p in cands:
            if pref in p.lower():
                return p
    return cands[0] if cands else None


def num_frame(df, colmap, cars, param):
    return pd.DataFrame(
        {c: pd.to_numeric(df[colmap[(c, param)]], errors="coerce") if (c, param) in colmap
         else np.nan for c in cars})


def str_frame(df, colmap, cars, param):
    return pd.DataFrame(
        {c: df[colmap[(c, param)]].astype("string") if (c, param) in colmap
         else pd.Series([pd.NA] * len(df), dtype="string") for c in cars})


def generic_residual_matrix(df, cars, colmap, params, cfg):
    # used when no indoor-temperature column exists (e.g. the ~60-parameter schema):
    # z-score every shared numeric parameter's deviation from the cross-car median at
    # each timestamp, then average across parameters. drops constant/categorical columns.
    mats, used = [], []
    for p in params:
        present = sum((c, p) in colmap for c in cars)
        if present < cfg["min_cars"]:
            continue
        M = num_frame(df, colmap, cars, p)
        if M.notna().sum().sum() == 0 or M.stack().nunique() < 3:
            continue
        dev = M.sub(M.median(axis=1), axis=0)
        std = dev.stack().std()
        if not std or np.isnan(std) or std == 0:
            continue
        mats.append(dev / std)
        used.append(p)
    if not mats:
        return None, "no usable generic numeric parameters found either"
    stacked = np.dstack([m[cars].values for m in mats])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        avg = np.nanmean(stacked, axis=2)
    res = pd.DataFrame(avg, index=mats[0].index, columns=cars)
    info = f"generic fallback over {len(used)} numeric parameters (no indoor-temperature column)"
    return res, info


def residual_matrix(df, cfg=DEFAULT_CFG):
    """Per-timestamp residual of each car's (indoor - cooling set-point) vs the train
    median, restricted to cooling mode and valid data. Returns (res, cars, info), or
    (None, cars, reason) if no usable signal can be built."""
    cars, colmap = parse_columns(df)
    if not cars:
        raise ValueError("no 'Car NN - <parameter>' columns found")
    params = sorted({k[1] for k in colmap})

    p_ind = find_param(params, ["indoor", "temp"], prefer=["average"])
    p_sp = (find_param(params, ["control", "temp", "cool"]) or
            find_param(params, ["target", "temp"]) or
            find_param(params, ["set", "temp", "cool"]))
    p_run = find_param(params, ["running mode"])
    p_val = find_param(params, ["information valid"])
    if p_ind is None:
        res, info = generic_residual_matrix(df, cars, colmap, params, cfg)
        if res is None:
            return None, cars, info
        return res, cars, info

    ind = num_frame(df, colmap, cars, p_ind)
    ind = ind.where(ind > 0)                                  # 0.0 = sentinel
    if p_sp is not None:
        sp = num_frame(df, colmap, cars, p_sp)
        sp = sp.where(sp > 0)
        err = ind - sp
    else:
        err = ind.copy()
    ok = pd.DataFrame(True, index=err.index, columns=cars)
    if p_run is not None:
        rm = str_frame(df, colmap, cars, p_run).apply(lambda s: s.str.lower())
        ok &= rm.apply(lambda s: s.str.contains("cool", na=False))
    if p_val is not None:
        vd = str_frame(df, colmap, cars, p_val).apply(lambda s: s.str.lower())
        ok &= ~vd.apply(lambda s: s.str.contains("invalid", na=False))
    err = err.where(ok)
    med = err.median(axis=1)
    enough = err.notna().sum(axis=1) >= cfg["min_cars"]
    res = err.sub(med, axis=0).where(enough, np.nan)
    info = f"indoor='{p_ind}' setpoint='{p_sp}' run='{p_run}' valid='{p_val}'"
    return res, cars, info


def car_features(res, cfg=DEFAULT_CFG):
    rows = {}
    for c in res.columns:
        r = res[c].dropna()
        if len(r) < cfg["min_samples"]:
            rows[c] = {f: np.nan for f in FEATURES}
            continue
        k = max(int(len(r) * cfg["late_frac"]), 1)
        rows[c] = {
            "res_mean": r.mean(),
            "res_p90": r.quantile(0.9),
            "frac_hot": (r > cfg["res_thresh"]).mean(),
            "late_mean": r.iloc[-k:].mean(),
        }
    return pd.DataFrame(rows).T[FEATURES]


def zscore_across_cars(F):
    mu, sd = F.mean(), F.std(ddof=0).replace(0, np.nan)
    Z = (F - mu) / sd
    return Z.fillna(-3.0) if len(F) else Z


def generic_scores(df):
    # last-resort fallback: mean |deviation from train median| of every numeric
    # parameter shared by all cars, z-scored per parameter
    cars, colmap = parse_columns(df)
    params = sorted({k[1] for k in colmap if all((c, k[1]) in colmap for c in cars)})
    zs = []
    for p in params:
        M = num_frame(df, colmap, cars, p)
        if M.notna().sum().sum() == 0 or M.stack().nunique() < 3:
            continue
        dev = M.sub(M.median(axis=1), axis=0).abs().mean()
        if dev.std(ddof=0) > 0:
            zs.append((dev - dev.mean()) / dev.std(ddof=0))
    if not zs:
        return pd.Series(0.0, index=cars)
    return pd.concat(zs, axis=1).mean(axis=1)


def load_model(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            m = json.load(fh)
        cfg = {**DEFAULT_CFG, **m.get("config", {})}
        return m["weights"], cfg
    print(f"model file '{path}' not found, using default weights", file=sys.stderr)
    return dict(DEFAULT_WEIGHTS), dict(DEFAULT_CFG)


def score_file_df(df, weights, cfg):
    """-> (pd.Series(score, index=car ids), info str). Higher score = more likely faulty."""
    res, cars, info = residual_matrix(df, cfg)
    if res is None:
        print(f"{info}, falling back to generic deviation score", file=sys.stderr)
        return generic_scores(df), f"last-resort generic_scores ({info})"
    F = car_features(res, cfg)
    Z = zscore_across_cars(F)
    w = np.array([weights.get(f, 0.0) for f in FEATURES], dtype=float)
    return pd.Series(Z.values @ w, index=Z.index), info


def rank_cars(score):
    return sorted(score.index, key=lambda c: (-float(score[c]), len(c), c))


def list_inputs(path):
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith((".xlsx", ".xls", ".csv")) and not f.startswith("~$"))
    return [path]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="test")
    ap.add_argument("--output", default="acv_predictions.csv")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "acv_model.json"))
    a = ap.parse_args()

    if not os.path.exists(a.input):
        sys.exit(f"'{a.input}' not found. Put test .xlsx file(s) in a folder called 'test', or pass --input <path>.")
    files = list_inputs(a.input)
    if not files:
        sys.exit(f"no .xlsx/.csv files found in '{a.input}'.")

    weights, cfg = load_model(a.model)
    out = []
    for p in files:
        try:
            df = read_table(p)
            sc, info = score_file_df(df, weights, cfg)
            ranked = rank_cars(sc)
        except Exception as e:
            print(f"{p}: {e}", file=sys.stderr)
            try:
                cars, _ = parse_columns(read_table(p))
            except Exception:
                cars = []
            ranked, info = cars, "error, see message above"
        print(f"{os.path.basename(p)} [{info}]")
        print(f"  top pick: {ranked[0] if ranked else '?'}  ranking: {'|'.join(ranked)}")
        out.append({"file_id": os.path.basename(p), "ranked_cars": "|".join(ranked)})
    pd.DataFrame(out, columns=["file_id", "ranked_cars"]).to_csv(a.output, sep="\t", index=False, lineterminator="\n")
    print(f"wrote {a.output} ({len(out)} row(s))")


if __name__ == "__main__":
    main()
