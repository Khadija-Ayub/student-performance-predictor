"""
preprocessor.py
────────────────────────────────────────────────────────────
Full preprocessing pipeline:
  - Missing value imputation
  - Outlier capping (IQR)
  - Encoding (ordinal + one-hot)
  - Feature scaling
  - Train/val/test split

Returns sklearn Pipeline objects that are saved as joblib
artifacts so the Streamlit app uses identical transformations.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import (
    StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
)
from sklearn.impute import SimpleImputer

# ── Column groups ─────────────────────────────────────────
DROP_COLS      = ["student_id"]
TARGET_COL     = "risk_level"

NUMERIC_COLS   = [
    "age", "prev_gpa", "matric_marks", "inter_marks",
    "failed_subjects", "backlogs", "attendance_pct",
    "study_hours_day", "library_visits", "lms_logins",
    "assignment_sub_pct", "midterm_score", "stress_level",
    "motivation", "sleep_hours", "commute_hours", "semester",
]

BINARY_COLS    = ["part_time_job", "scholarship"]   # already 0/1

ORDINAL_COLS   = ["family_income", "father_education", "mother_education"]
ORDINAL_CATS   = [
    ["Low", "Middle", "High"],
    ["None", "Primary", "Secondary", "Graduate", "Postgrad"],
    ["None", "Primary", "Secondary", "Graduate", "Postgrad"],
]

NOMINAL_COLS   = ["gender", "residence", "department"]

LABEL_ORDER    = ["Low Risk", "Medium Risk", "High Risk"]


def cap_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Cap outliers at 1.5 × IQR bounds."""
    df = df.copy()
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build and return the sklearn ColumnTransformer."""

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ORDINAL_CATS,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])

    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_COLS),
        ("ordinal", ordinal_pipe, ORDINAL_COLS),
        ("nominal", nominal_pipe, NOMINAL_COLS),
        ("binary",  binary_pipe,  BINARY_COLS),
    ], remainder="drop")

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract feature names after fit."""
    names = []
    names.extend(NUMERIC_COLS)
    names.extend(ORDINAL_COLS)
    # One-hot encoded
    ohe = preprocessor.named_transformers_["nominal"]["encoder"]
    names.extend(ohe.get_feature_names_out(NOMINAL_COLS).tolist())
    names.extend(BINARY_COLS)
    return names


def run_preprocessing(
    raw_path: str  = "data/raw/students.csv",
    save_dir: str  = "data/processed",
    artifact_dir: str = "models",
):
    save_dir     = Path(save_dir)
    artifact_dir = Path(artifact_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────
    df = pd.read_csv(raw_path)
    print(f"Loaded  : {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Drop unused ───────────────────────────────────────
    df.drop(columns=DROP_COLS, inplace=True)

    # ── Cap outliers in numeric cols ──────────────────────
    df = cap_outliers(df, NUMERIC_COLS)

    # ── Encode target ─────────────────────────────────────
    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    y = le.transform(df[TARGET_COL])
    X = df.drop(columns=[TARGET_COL])

    # ── Split ─────────────────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train   : {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── Fit preprocessor on train only ────────────────────
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)
    X_test_proc  = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    # ── Save processed arrays ─────────────────────────────
    np.save(save_dir / "X_train.npy", X_train_proc)
    np.save(save_dir / "X_val.npy",   X_val_proc)
    np.save(save_dir / "X_test.npy",  X_test_proc)
    np.save(save_dir / "y_train.npy", y_train)
    np.save(save_dir / "y_val.npy",   y_val)
    np.save(save_dir / "y_test.npy",  y_test)

    # Save feature names
    pd.Series(feature_names).to_csv(save_dir / "feature_names.csv", index=False, header=False)

    # ── Save artifacts ────────────────────────────────────
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    joblib.dump(le,           artifact_dir / "label_encoder.joblib")

    print(f"✓ Preprocessed data  → {save_dir}")
    print(f"✓ Artifacts saved    → {artifact_dir}")
    print(f"  Features           : {len(feature_names)}")

    return X_train_proc, X_val_proc, X_test_proc, y_train, y_val, y_test, feature_names


if __name__ == "__main__":
    run_preprocessing()
