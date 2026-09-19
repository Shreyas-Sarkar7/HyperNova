"""Predict Normal / Side I / Side II for rail-corrugation CSVs.

  python predict.py --input path/to/csv_dir_or_file --output rail_predictions.csv
Optional, AFTER predicting:  --labels labels.csv   (columns file_id,label) to score the predictions.
"""
import argparse, warnings
import joblib, numpy as np, pandas as pd
import features as ft
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", default="rail_predictions.csv")
ap.add_argument("--model", default="model.joblib")
ap.add_argument("--proba-output", default=None)
ap.add_argument("--labels", default=None, help="optional reference labels (file_id,label) to score predictions")
a = ap.parse_args()

bundle = joblib.load(a.model)
feats, _ = ft.extract_dir(ft.list_csvs(a.input), verbose=False)
X = feats.reindex(columns=bundle["feature_names"])
pred = bundle["model"].predict(X)
out = pd.DataFrame({"file_id": feats["file_id"], "prediction": pred})
out.to_csv(a.output, index=False)
print(f"wrote {a.output} ({len(out)} rows)"); print(out["prediction"].value_counts().to_string())
if a.proba_output:
    P = pd.DataFrame(bundle["model"].predict_proba(X), columns=bundle["model"].classes_)
    P.insert(0, "file_id", feats["file_id"]); P.round(4).to_csv(a.proba_output, index=False)
if a.labels:
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    lab = pd.read_csv(a.labels).merge(out, on="file_id")
    C = bundle["classes"]
    print("\nn =", len(lab), " accuracy %.3f  macro-F1 %.3f" % (accuracy_score(lab.label, lab.prediction),
          f1_score(lab.label, lab.prediction, labels=C, average="macro", zero_division=0)))
    print(classification_report(lab.label, lab.prediction, labels=C, zero_division=0))
    print(pd.DataFrame(confusion_matrix(lab.label, lab.prediction, labels=C), index=C, columns=C))
