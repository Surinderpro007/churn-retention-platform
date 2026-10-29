"""Train the churn model, compare candidates with cross-validation and save everything.

Run from the project root:  python -m scripts.train_model

Outputs (in model/):
  churn_model.joblib  - the chosen sklearn pipeline
  metrics.json        - test-set metrics + cross-validation comparison
  eval_data.csv       - test labels and predicted probabilities (used by the Model Report tab)
"""
import json
from datetime import datetime

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.config import CAT, FEATURES, NUM, TARGET
from src.metrics import best_f1_threshold, metrics_at_threshold

LOGREG = "Logistic Regression"
# Prefer the simple, explainable model unless another one is clearly better.
MIN_AUC_GAIN_TO_SWITCH = 0.005


def build_pipeline(clf) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def candidates() -> dict:
    # class_weight="balanced" gives the (rarer) churn class more weight -> better recall.
    return {
        LOGREG: build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": build_pipeline(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1)),
        "Gradient Boosting": build_pipeline(HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced",
            random_state=config.RANDOM_STATE)),
    }


def main():
    df = pd.read_csv(config.DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    # 80/20 split; stratify keeps the churn ratio equal in train and test.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    # 1) Compare candidate models with 5-fold cross-validation on the TRAIN set only.
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    models, cv_results = candidates(), []
    for name, pipe in models.items():
        s = cross_validate(pipe, Xtr, ytr, cv=cv, scoring={"roc_auc": "roc_auc", "f1": "f1", "recall": "recall"})
        cv_results.append({"model": name,
                           "cv_roc_auc": float(s["test_roc_auc"].mean()),
                           "cv_f1": float(s["test_f1"].mean()),
                           "cv_recall": float(s["test_recall"].mean())})
        print(f"{name:20s} CV ROC-AUC {cv_results[-1]['cv_roc_auc']:.3f} | "
              f"F1 {cv_results[-1]['cv_f1']:.3f} | recall {cv_results[-1]['cv_recall']:.3f}")

    # 2) Pick the winner (simple model wins unless clearly beaten).
    best = max(cv_results, key=lambda r: r["cv_roc_auc"])
    base = next(r for r in cv_results if r["model"] == LOGREG)
    chosen = LOGREG if best["cv_roc_auc"] - base["cv_roc_auc"] < MIN_AUC_GAIN_TO_SWITCH else best["model"]
    print(f"-> chosen model: {chosen}")

    # 3) Fit the winner on the train set and evaluate once on the untouched test set.
    model = models[chosen].fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    m = metrics_at_threshold(yte, prob, 0.5)
    m.pop("threshold")
    m["roc_auc"] = float(roc_auc_score(yte, prob))
    m["best_f1_threshold"], m["best_f1"] = best_f1_threshold(yte, prob)
    m.update({
        "model_name": chosen,
        "cv_results": cv_results,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sklearn_version": sklearn.__version__,
    })

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(m, indent=2))
    pd.DataFrame({"y_true": yte.to_numpy(), "y_prob": prob}).to_csv(config.EVAL_PATH, index=False)
    print(json.dumps({k: v for k, v in m.items() if k != "cv_results"}, indent=2))


if __name__ == "__main__":
    main()
