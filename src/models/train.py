"""
train.py
────────────────────────────────────────────────────────────
Trains 3 classifiers and saves the best model:
  1. Logistic Regression  (baseline)
  2. Random Forest        (ensemble)
  3. XGBoost              (gradient boosting)

Hyperparameter tuning via RandomizedSearchCV on val set.
Best model selected by macro F1 score.
All artifacts saved to models/.
"""

import numpy as np
import pandas as pd
import joblib
import json
import warnings
from pathlib import Path
from time import time

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not installed. Skipping XGBoost model.")

LABEL_NAMES = ["Low Risk", "Medium Risk", "High Risk"]
DATA_DIR    = Path("data/processed")
MODEL_DIR   = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")
    y_test  = np.load(DATA_DIR / "y_test.npy")

    # Combine train+val for final model fitting
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    features = pd.read_csv(DATA_DIR / "feature_names.csv", header=None)[0].tolist()
    return X_train, X_val, X_test, y_train, y_val, y_test, X_trainval, y_trainval, features


def evaluate(model, X, y, split_name=""):
    y_pred = model.predict(X)
    metrics = {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "f1_macro":  round(f1_score(y, y_pred, average="macro"), 4),
        "precision": round(precision_score(y, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, average="macro"), 4),
    }
    if split_name:
        print(f"  [{split_name}] Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1_macro']:.4f}  "
              f"Prec={metrics['precision']:.4f}  "
              f"Rec={metrics['recall']:.4f}")
    return metrics, y_pred


def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("\n── Logistic Regression ──────────────────────────────")
    param_dist = {
        "C":            [0.01, 0.1, 1.0, 5.0, 10.0],
        "solver":       ["lbfgs", "saga"],
        "max_iter":     [500, 1000],
        "class_weight": [None, "balanced"],
    }
    base = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=20, cv=cv,
        scoring="f1_macro", random_state=42, n_jobs=-1, verbose=0
    )
    t0 = time()
    search.fit(X_train, y_train)
    print(f"  Best params : {search.best_params_}")
    print(f"  Train time  : {time()-t0:.1f}s")
    best = search.best_estimator_
    val_metrics, _ = evaluate(best, X_val, y_val, "Val")
    return best, val_metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n── Random Forest ────────────────────────────────────")
    param_dist = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 8, 12, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
        "class_weight":      [None, "balanced"],
    }
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=25, cv=cv,
        scoring="f1_macro", random_state=42, n_jobs=-1, verbose=0
    )
    t0 = time()
    search.fit(X_train, y_train)
    print(f"  Best params : {search.best_params_}")
    print(f"  Train time  : {time()-t0:.1f}s")
    best = search.best_estimator_
    val_metrics, _ = evaluate(best, X_val, y_val, "Val")
    return best, val_metrics


def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n── XGBoost ──────────────────────────────────────────")
    if not XGB_AVAILABLE:
        print("  Skipped (not installed)")
        return None, None

    param_dist = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [3, 5, 7, 9],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "gamma":           [0, 0.1, 0.3],
        "reg_alpha":       [0, 0.1, 0.5],
        "reg_lambda":      [1, 1.5, 2],
    }
    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=30, cv=cv,
        scoring="f1_macro", random_state=42, n_jobs=-1, verbose=0
    )
    t0 = time()
    search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best params : {search.best_params_}")
    print(f"  Train time  : {time()-t0:.1f}s")
    best = search.best_estimator_
    val_metrics, _ = evaluate(best, X_val, y_val, "Val")
    return best, val_metrics


def run_training():
    print("=" * 60)
    print("  Student Performance Predictor — Model Training")
    print("=" * 60)

    (X_train, X_val, X_test, y_train, y_val, y_test,
     X_trainval, y_trainval, features) = load_data()

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ── Train all models ──────────────────────────────────
    lr_model,  lr_metrics  = train_logistic_regression(X_train, y_train, X_val, y_val)
    rf_model,  rf_metrics  = train_random_forest(X_train, y_train, X_val, y_val)
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)

    # ── Compare and select best ───────────────────────────
    candidates = {
        "logistic_regression": (lr_model,  lr_metrics),
        "random_forest":       (rf_model,  rf_metrics),
    }
    if xgb_model is not None:
        candidates["xgboost"] = (xgb_model, xgb_metrics)

    print("\n── Model Comparison (Validation F1 Macro) ───────────")
    results = {}
    for name, (model, metrics) in candidates.items():
        print(f"  {name:25s}  F1={metrics['f1_macro']:.4f}  Acc={metrics['accuracy']:.4f}")
        results[name] = metrics

    best_name  = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = candidates[best_name][0]
    print(f"\n  ★  Best model: {best_name}")

    # ── Refit best model on train+val ─────────────────────
    best_model.fit(X_trainval, y_trainval)

    # ── Final test evaluation ─────────────────────────────
    print("\n── Final Test Set Evaluation ────────────────────────")
    test_metrics, y_pred = evaluate(best_model, X_test, y_test, "Test")
    print("\n" + classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    # ── Save artifacts ────────────────────────────────────
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(lr_model,   MODEL_DIR / "logistic_regression.joblib")
    joblib.dump(rf_model,   MODEL_DIR / "random_forest.joblib")
    if xgb_model:
        joblib.dump(xgb_model, MODEL_DIR / "xgboost.joblib")

    # Save results summary
    summary = {
        "best_model":     best_name,
        "val_results":    results,
        "test_results":   test_metrics,
        "feature_count":  len(features),
        "label_names":    LABEL_NAMES,
    }
    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save feature importances (RF or XGB)
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature":   features,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
        fi.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
        print(f"\n  Top 10 features:\n{fi.head(10).to_string(index=False)}")

    print(f"\n✓ All artifacts saved → {MODEL_DIR}")
    return best_model, best_name, test_metrics


if __name__ == "__main__":
    run_training()
