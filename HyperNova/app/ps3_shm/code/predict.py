import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd

from features import load_signal, miner_proxy, extract_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing test CSVs")
    parser.add_argument("--output", required=True, help="Path to shm_predictions.csv")
    args = parser.parse_args()

    with open("../model/config.json") as f:
        config = json.load(f)

    if config["model_type"] == "power_law":
        m = config["m"]
        alpha = config["alpha"]
        beta = config["beta"]
        model = None
    else:
        model = joblib.load("../model/model.joblib")

    rows = []
    for fname in sorted(os.listdir(args.input)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(args.input, fname)
        signal = load_signal(path)

        if config["model_type"] == "power_law":
            proxy = miner_proxy(signal, m)
            pred = alpha * (proxy ** beta)
        else:
            feats = extract_features(signal)
            X = pd.DataFrame([feats])[config["feature_columns"]]
            pred_log = model.predict(X)[0]
            pred = np.exp(pred_log)

        pred = max(float(pred), 1e-12)
        rows.append({"file_id": fname, "prediction": pred})

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} predictions to {args.output}")

if __name__ == "__main__":
    main()