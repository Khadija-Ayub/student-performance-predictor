"""
feature_engineering.py
────────────────────────────────────────────────────────────
Creates domain-driven derived features BEFORE preprocessing.
Run this on the raw CSV to produce an enriched CSV.

Domain knowledge applied:
  - Academic momentum index
  - Engagement score
  - Support index
  - Risk composite (for EDA only, not used in model)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 1. Academic Momentum Index ────────────────────────
    # Combines GPA trend + midterm performance + prior marks
    df["academic_momentum"] = (
        df["prev_gpa"] / 4.0 * 40 +
        df["midterm_score"] / 100 * 40 +
        (df["matric_marks"] + df["inter_marks"]) / 200 * 20
    ).round(2)

    # ── 2. Engagement Score (0–100) ───────────────────────
    df["engagement_score"] = (
        df["attendance_pct"]    * 0.35 +
        df["assignment_sub_pct"]* 0.30 +
        (df["lms_logins"] / df["lms_logins"].max()) * 100 * 0.20 +
        (df["library_visits"] / df["library_visits"].max()) * 100 * 0.15
    ).round(2)

    # ── 3. Study Efficiency ───────────────────────────────
    # High study hours + low GPA = inefficient (potential learning issue)
    df["study_efficiency"] = (
        df["prev_gpa"] / (df["study_hours_day"].fillna(1) + 0.5)
    ).round(3)

    # ── 4. Support Index ──────────────────────────────────
    # Family support proxy: income + parental education + scholarship
    edu_map = {"None": 0, "Primary": 1, "Secondary": 2, "Graduate": 3, "Postgrad": 4}
    inc_map = {"Low": 0, "Middle": 1, "High": 2}

    df["support_index"] = (
        df["family_income"].map(inc_map).fillna(0) * 2 +
        df["father_education"].map(edu_map).fillna(0) +
        df["mother_education"].map(edu_map).fillna(0) +
        df["scholarship"] * 2
    ).round(2)

    # ── 5. Stress-Motivation Balance ─────────────────────
    # Positive = motivated despite stress, Negative = demotivated
    df["stress_motivation_ratio"] = (
        df["motivation"] - df["stress_level"]
    )

    # ── 6. Lifestyle Score ────────────────────────────────
    # Higher = healthier lifestyle
    df["lifestyle_score"] = (
        df["sleep_hours"] / 8 * 40 +                        # ideal = 8hrs
        (1 - df["part_time_job"]) * 20 +
        np.clip(8 - df["commute_hours"], 0, 8) / 8 * 20 +
        df["study_hours_day"].fillna(0) / 8 * 20
    ).round(2)

    # ── 7. Academic Burden ───────────────────────────────
    df["academic_burden"] = (
        df["failed_subjects"] * 3 +
        df["backlogs"] * 2 +
        df["stress_level"]
    )

    # ── 8. GPA Band (ordinal binning) ────────────────────
    df["gpa_band"] = pd.cut(
        df["prev_gpa"],
        bins=[0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        labels=["F", "D", "C", "B", "B+", "A"],
        right=True
    ).astype(str)

    # ── 9. Attendance Category ───────────────────────────
    df["attendance_category"] = pd.cut(
        df["attendance_pct"],
        bins=[0, 50, 65, 75, 85, 100],
        labels=["Critical", "Poor", "Average", "Good", "Excellent"],
        right=True
    ).astype(str)

    # ── 10. Semesters at risk flag ────────────────────────
    df["is_early_semester"] = (df["semester"] <= 2).astype(int)

    return df


def run_feature_engineering(
    raw_path: str    = "data/raw/students.csv",
    output_path: str = "data/raw/students_engineered.csv",
):
    df = pd.read_csv(raw_path)
    print(f"Input shape  : {df.shape}")

    df_eng = engineer_features(df)
    df_eng.to_csv(output_path, index=False)

    new_cols = [c for c in df_eng.columns if c not in df.columns]
    print(f"Output shape : {df_eng.shape}")
    print(f"New features : {new_cols}")
    print(f"✓ Saved → {output_path}")

    return df_eng


if __name__ == "__main__":
    run_feature_engineering()
