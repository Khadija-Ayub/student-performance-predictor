"""
evaluate.py
────────────────────────────────────────────────────────────
Comprehensive model evaluation:
  - Per-class metrics
  - Confusion matrix heatmap
  - ROC-AUC curves (one-vs-rest)
  - Precision-Recall curves
  - Model comparison bar charts
  - Saves all figures to reports/figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize

LABEL_NAMES  = ["Low Risk", "Medium Risk", "High Risk"]
COLORS       = ["#10b981", "#f59e0b", "#ef4444"]   # green, amber, red
FIGURES_DIR  = Path("reports/figures")
MODEL_DIR    = Path("models")
DATA_DIR     = Path("data/processed")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":  "#0f111a",
    "axes.facecolor":    "#1c2033",
    "axes.edgecolor":    "#2d3250",
    "axes.labelcolor":   "#e2e8f0",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "text.color":        "#e2e8f0",
    "grid.color":        "#2d3250",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
})


def load_artifacts():
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    return model, X_test, y_test


def plot_confusion_matrix(y_true, y_pred, save=True):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% of True Class")

    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(LABEL_NAMES)

    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_pct[i,j] > 50 else "#94a3b8",
                    fontsize=11, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, pad=15)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        print("  Saved: confusion_matrix.png")
    return fig


def plot_roc_curves(model, X_test, y_test, save=True):
    y_bin  = label_binarize(y_test, classes=[0, 1, 2])
    y_prob = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (label, color) in enumerate(zip(LABEL_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — One vs Rest", fontsize=14, pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
        print("  Saved: roc_curves.png")
    return fig


def plot_precision_recall(model, X_test, y_test, save=True):
    y_bin  = label_binarize(y_test, classes=[0, 1, 2])
    y_prob = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (label, color) in enumerate(zip(LABEL_NAMES, COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{label}  (AP = {ap:.3f})")

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "precision_recall.png", dpi=150, bbox_inches="tight")
        print("  Saved: precision_recall.png")
    return fig


def plot_model_comparison(save=True):
    """Bar chart comparing all 3 models on key metrics."""
    import json
    with open(MODEL_DIR / "training_summary.json") as f:
        summary = json.load(f)

    val_results = summary["val_results"]
    models  = list(val_results.keys())
    metrics = ["accuracy", "f1_macro", "precision", "recall"]
    labels  = ["Accuracy", "F1 Macro", "Precision", "Recall"]

    x     = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_colors = ["#6366f1", "#10b981", "#f59e0b"]
    for i, (model_name, color) in enumerate(zip(models, bar_colors)):
        vals = [val_results[model_name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model_name.replace("_", " ").title(),
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#94a3b8")

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Validation Set", fontsize=14, pad=15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
        print("  Saved: model_comparison.png")
    return fig


def plot_feature_importance(top_n=20, save=True):
    fi_path = MODEL_DIR / "feature_importance.csv"
    if not fi_path.exists():
        print("  Feature importance file not found — skipping.")
        return

    fi = pd.read_csv(fi_path).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))

    bars = ax.barh(fi["feature"][::-1], fi["importance"][::-1],
                   color="#6366f1", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, pad=15)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=9)

    for bar, val in zip(bars, fi["importance"][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#94a3b8")
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        print("  Saved: feature_importance.png")
    return fig


def run_evaluation():
    print("=" * 60)
    print("  Model Evaluation")
    print("=" * 60)

    model, X_test, y_test = load_artifacts()
    y_pred = model.predict(X_test)

    print("\n── Classification Report ────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    print("── Generating figures ───────────────────────────────")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curves(model, X_test, y_test)
    plot_precision_recall(model, X_test, y_test)
    plot_model_comparison()
    plot_feature_importance()

    print(f"\n✓ All figures saved → {FIGURES_DIR}")


if __name__ == "__main__":
    run_evaluation()
