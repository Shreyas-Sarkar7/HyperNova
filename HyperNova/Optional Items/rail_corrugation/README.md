# Rail Corrugation (NebulaX PS3) - final model, trained on Train1-Train272

Model: HistGradientBoosting (depth 3, 150 iterations, balanced class weights) on 279 features from the vibration
channels (per axle box time/frequency/wavelength features, aggregated over Side I boxes = positions 1,3,5,7 and
Side II boxes = positions 2,4,6,8, plus Side I - Side II contrasts and 3 speed features).
The model type was chosen with repeated and nested CV on Train1-250 only. Train251-272 were then predicted
as an untouched hold-out (20/22 correct) and added to training afterwards.

Performance estimate (no unseen data is left): repeated stratified 5-fold CV on the 272 files gives
macro F1 0.835 +/- 0.020, accuracy 0.952, per-class F1 (Normal / Side I / Side II) 0.98 / 0.66 / 0.87.
Side I has only 14 training files, so it is the weakest class and its estimate is noisy.

## Predict (e.g. for the official test set)
    pip install -r requirements.txt
    python predict.py --input <folder_of_csvs> --output rail_predictions.csv
Output columns: file_id,prediction. Add --proba-output probs.csv for class probabilities, or
--labels labels.csv (file_id,label) to score predictions. Check the exact command-line format required by the
hackathon spec; this script's --input/--output interface was not verified against it.

## Rebuild from scratch
    python features.py --input <folder_with_Train1-272_csvs> --output features_train1_272.csv
    python train_final.py       # CV estimate + fit on all 272 files -> model.joblib, cv_results.json

Files: features.py (feature extraction), models.py, train_final.py, predict.py, model.joblib,
features_train1_272.csv (cached features), labels_train1_272.csv, cv_results.json, requirements.txt.
