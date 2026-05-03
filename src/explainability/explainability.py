"""
explainability.py
────────────────────────────────────────────────────────────
SHAP-based model explainability:
  - Global feature importance (beeswarm + bar)
  - Waterfall plot for single prediction
  - SHAP summary per class
  - Saves plots to reports/figures/

Also provides explain_prediction() used by the Streamlit app.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠  SHAP not installed. Run: pip install shap")

LABEL_NAMES = ["Low Risk", "Medium Risk", "High Risk"]
MODEL_DIR   = Path("models")
DATA_DIR    = Path("data/processed")
FIG_DIR     = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f111a",
    "axes.facecolor":   "#1c2033",
    "text.color":       "#e2e8f0",
    "axes.labelcolor":  "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
})


def get_explainer(model, X_background: np.ndarray):
    """Return appropriate SHAP explainer for given model type."""
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier",
                      "GradientBoostingClassifier", "DecisionTreeClassifier"):
        return shap.TreeExplainer(model)
    else:
        background = shap.sample(X_background, 100, random_state=42)
        return shap.KernelExplainer(model.predict_proba, background)


def plot_shap_summary(explainer, X_sample, feature_names, class_idx=2, save=True):
    """Global SHAP beeswarm for a given class (default: High Risk)."""
    shap_values = explainer.shap_values(X_sample)

    # Handle multi-output formats
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    else:
        sv = shap_values[:, :, class_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv, X_sample,
        feature_names=feature_names,
        show=False, plot_type="dot",
        max_display=20, color_bar=True,
    )
    plt.title(f"SHAP Summary — {LABEL_NAMES[class_idx]}", fontsize=14, pad=15)
    plt.tight_layout()

    if save:
        fname = FIG_DIR / f"shap_summary_class{class_idx}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0f111a")
        print(f"  Saved: {fname.name}")
    plt.close()


def plot_shap_bar(explainer, X_sample, feature_names, save=True):
    """Mean absolute SHAP bar chart (global)."""
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=(0, 2))

    fi = pd.DataFrame({"feature": feature_names, "mean_shap": mean_shap})
    fi = fi.sort_values("mean_shap", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(fi["feature"], fi["mean_shap"], color="#6366f1", alpha=0.85, edgecolor="white", lw=0.4)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=14, pad=15)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()

    if save:
        fname = FIG_DIR / "shap_bar_global.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0f111a")
        print(f"  Saved: {fname.name}")
    plt.close(fig)


def explain_prediction(
    model,
    preprocessor,
    input_df: pd.DataFrame,
    feature_names: list,
    X_background: np.ndarray,
) -> dict:
    """
    Explain a single student prediction.
    Returns dict with top positive and negative SHAP drivers.
    Used by the Streamlit app.
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP not installed"}

    X_proc = preprocessor.transform(input_df)
    explainer   = get_explainer(model, X_background)
    shap_values = explainer.shap_values(X_proc)

    pred_class = int(model.predict(X_proc)[0])
    prob       = model.predict_proba(X_proc)[0]

    if isinstance(shap_values, list):
        sv = shap_values[pred_class][0]
    else:
        sv = shap_values[0, :, pred_class]

    # Build feature-shap pairs
    pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)

    top_positive = [(f, round(float(v), 4)) for f, v in pairs if v > 0][:5]
    top_negative = [(f, round(float(v), 4)) for f, v in pairs if v < 0][:5]

    return {
        "predicted_class":   pred_class,
        "predicted_label":   LABEL_NAMES[pred_class],
        "probabilities":     {LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(prob)},
        "top_risk_factors":  top_positive,
        "top_protectors":    top_negative,
    }


def run_explainability():
    if not SHAP_AVAILABLE:
        print("SHAP not installed — skipping. Install with: pip install shap")
        return

    print("=" * 60)
    print("  SHAP Explainability")
    print("=" * 60)

    model    = joblib.load(MODEL_DIR / "best_model.joblib")
    X_train  = np.load(DATA_DIR / "X_train.npy")
    X_test   = np.load(DATA_DIR / "X_test.npy")
    features = pd.read_csv(DATA_DIR / "feature_names.csv", header=None)[0].tolist()

    # Use 200 test samples for speed
    X_sample = X_test[:200]
    explainer = get_explainer(model, X_train)

    print("  Computing SHAP values (this may take a minute)…")
    plot_shap_bar(explainer, X_sample, features)
    for class_idx in range(3):
        plot_shap_summary(explainer, X_sample, features, class_idx=class_idx)

    print(f"\n✓ SHAP figures saved → {FIG_DIR}")


if __name__ == "__main__":
    run_explainability()
