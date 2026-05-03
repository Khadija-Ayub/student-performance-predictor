"""
run_pipeline.py
────────────────────────────────────────────────────────────
Master script — runs the complete ML pipeline end-to-end:
  1. Generate dataset
  2. Feature engineering
  3. Preprocessing
  4. Model training
  5. Evaluation
  6. SHAP explainability

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-shap    (faster, skips SHAP)
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main(skip_shap: bool = False):
    print("\n" + "=" * 60)
    print("  STUDENT PERFORMANCE PREDICTOR — Full Pipeline")
    print("=" * 60 + "\n")

    # ── Step 1: Generate Dataset ──────────────────────────
    print("STEP 1/6  Generating synthetic dataset…")
    from src.data.generate_dataset import generate_dataset
    import pandas as pd
    from pathlib import Path

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/raw/students.csv", index=False)
    print(f"  ✓ {len(df)} student records generated\n")

    # ── Step 2: Feature Engineering ───────────────────────
    print("STEP 2/6  Engineering features…")
    from src.features.feature_engineering import run_feature_engineering
    run_feature_engineering(
        raw_path="data/raw/students.csv",
        output_path="data/raw/students_engineered.csv"
    )
    print()

    # ── Step 3: Preprocessing ─────────────────────────────
    print("STEP 3/6  Preprocessing & splitting data…")
    from src.data.preprocessor import run_preprocessing
    run_preprocessing(
        raw_path="data/raw/students_engineered.csv",
        save_dir="data/processed",
        artifact_dir="models"
    )
    print()

    # ── Step 4: Train Models ──────────────────────────────
    print("STEP 4/6  Training models with hyperparameter tuning…")
    from src.models.train import run_training
    run_training()
    print()

    # ── Step 5: Evaluate ──────────────────────────────────
    print("STEP 5/6  Evaluating models & generating plots…")
    from src.evaluation.evaluate import run_evaluation
    run_evaluation()
    print()

    # ── Step 6: SHAP ──────────────────────────────────────
    if not skip_shap:
        print("STEP 6/6  Generating SHAP explanations…")
        from src.explainability.explainability import run_explainability
        run_explainability()
    else:
        print("STEP 6/6  SHAP skipped (--skip-shap flag)")

    print("\n" + "=" * 60)
    print("  ✓ Pipeline complete!")
    print("  → Model artifacts : models/")
    print("  → Figures         : reports/figures/")
    print("  → Run the app     : streamlit run app/streamlit_app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP explainability (speeds up pipeline)")
    args = parser.parse_args()
    main(skip_shap=args.skip_shap)
