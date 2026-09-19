"""Final model: HistGradientBoosting (selected by repeated + nested CV on Train1-250)."""
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

CLASSES = ["Normal", "Side I", "Side II"]


def final_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),   # fitted on training data only (zero-speed files have no wavelength features)
        HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0,
                                       min_samples_leaf=5, class_weight="balanced", random_state=0))
