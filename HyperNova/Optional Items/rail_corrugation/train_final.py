"""Train the final model on all 272 labelled files (Train1-Train272) and save model.joblib.
Also prints repeated stratified 5-fold CV (5 repeats) as the performance estimate."""
import json, warnings, joblib, sklearn
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
from models import final_model, CLASSES
warnings.filterwarnings("ignore")

X = pd.read_csv("features_train1_272.csv"); L = pd.read_csv("labels_train1_272.csv")
d = X.merge(L, on="file_id"); assert len(d) == 272
y = d["label"].to_numpy(); F = d.drop(columns=["file_id", "label"])
print("shape", F.shape, pd.Series(y).value_counts().to_dict())

f1s, accs, cm, prf = [], [], 0, []
for seed in range(5):
    oof = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, y):
        m = final_model().fit(F.iloc[tr], y[tr]); oof[te] = m.predict(F.iloc[te])
    f1s.append(f1_score(y, oof, labels=CLASSES, average="macro")); accs.append(accuracy_score(y, oof))
    cm = cm + confusion_matrix(y, oof, labels=CLASSES)
    prf.append(precision_recall_fscore_support(y, oof, labels=CLASSES)[:3])
prf = np.mean(prf, axis=0).round(3)
res = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s)), "accuracy": float(np.mean(accs)),
       "precision": prf[0].tolist(), "recall": prf[1].tolist(), "f1": prf[2].tolist(),
       "confusion_summed_5_repeats": cm.tolist(), "class_order": CLASSES}
json.dump(res, open("cv_results.json", "w"), indent=1)
print(f"CV macro F1 {res['macro_f1_mean']:.3f} ± {res['macro_f1_sd']:.3f}  acc {res['accuracy']:.3f}")
print("precision", res["precision"], "recall", res["recall"], "F1", res["f1"]); print(cm)

model = final_model().fit(F, y)
joblib.dump({"model": model, "feature_names": list(F.columns), "classes": CLASSES,
             "trained_on": "Train1.csv-Train272.csv", "sklearn_version": sklearn.__version__}, "model.joblib")
print("saved model.joblib")
