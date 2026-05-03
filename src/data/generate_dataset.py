"""
generate_dataset.py
────────────────────────────────────────────────────────────
Generates a realistic synthetic student performance dataset.
Inspired by the UCI Student Performance dataset structure but
extended with Pakistani university context.

Run:  python src/data/generate_dataset.py
Output: data/raw/students.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N    = 2000      # number of students

rng  = np.random.default_rng(SEED)


def generate_dataset(n: int = N) -> pd.DataFrame:
    # ── Demographic features ─────────────────────────────
    gender        = rng.choice(["Male", "Female"], n, p=[0.55, 0.45])
    age           = rng.integers(17, 26, n)
    residence     = rng.choice(["Urban", "Rural", "Suburban"], n, p=[0.50, 0.30, 0.20])
    family_income = rng.choice(["Low", "Middle", "High"], n, p=[0.35, 0.45, 0.20])
    father_edu    = rng.choice(["None","Primary","Secondary","Graduate","Postgrad"], n,
                               p=[0.10, 0.15, 0.30, 0.30, 0.15])
    mother_edu    = rng.choice(["None","Primary","Secondary","Graduate","Postgrad"], n,
                               p=[0.15, 0.20, 0.30, 0.25, 0.10])

    # ── Academic history ──────────────────────────────────
    prev_gpa         = np.clip(rng.normal(2.8, 0.7, n), 0.0, 4.0).round(2)
    matric_marks     = np.clip(rng.normal(72, 14, n), 33, 100).round(1)
    inter_marks      = np.clip(rng.normal(68, 15, n), 33, 100).round(1)
    failed_subjects  = rng.choice([0,1,2,3,4], n, p=[0.55, 0.25, 0.12, 0.05, 0.03])
    backlogs         = failed_subjects.copy()

    # ── Engagement & behavior ─────────────────────────────
    attendance_pct   = np.clip(rng.normal(76, 15, n), 10, 100).round(1)
    study_hours_day  = np.clip(rng.normal(3.2, 1.5, n), 0, 12).round(1)
    library_visits   = rng.integers(0, 30, n)           # per month
    lms_logins       = rng.integers(0, 60, n)            # per month
    assignment_sub   = np.clip(rng.normal(80, 18, n), 0, 100).round(1)  # % submitted

    # ── Psychosocial factors ──────────────────────────────
    stress_level     = rng.integers(1, 11, n)            # 1–10 scale
    motivation       = rng.integers(1, 11, n)
    sleep_hours      = np.clip(rng.normal(6.5, 1.2, n), 3, 10).round(1)
    part_time_job    = rng.choice([0, 1], n, p=[0.65, 0.35])
    commute_hours    = np.clip(rng.normal(1.2, 0.9, n), 0, 5).round(1)

    # ── Mid-term exam scores ──────────────────────────────
    midterm_score    = np.clip(rng.normal(55, 18, n), 0, 100).round(1)

    # ── Department ───────────────────────────────────────
    department       = rng.choice(
        ["CS", "SE", "IT", "EE", "ME", "BBA", "Economics"],
        n, p=[0.22, 0.18, 0.15, 0.15, 0.10, 0.12, 0.08]
    )
    semester         = rng.integers(1, 9, n)

    # ── Scholarships / financial aid ──────────────────────
    scholarship      = rng.choice([0, 1], n, p=[0.72, 0.28])

    # ── Target variable: RISK LEVEL ───────────────────────
    # Compute a weighted risk score — higher score = higher risk
    risk_score = (
        (4.0 - prev_gpa)          * 2.0   +
        (100 - midterm_score)      * 0.04  +
        (100 - attendance_pct)     * 0.03  +
        (10  - motivation)         * 0.25  +
        stress_level               * 0.20  +
        failed_subjects            * 0.50  +
        (10  - study_hours_day)    * 0.15  +
        part_time_job              * 0.30  +
        (family_income == "Low").astype(int) * 0.40 +
        rng.normal(0, 0.3, n)              # noise
    )

    # Map score → 3-class risk label
    low_thresh  = np.percentile(risk_score, 45)
    high_thresh = np.percentile(risk_score, 75)

    risk_label = np.where(
        risk_score <= low_thresh,  "Low Risk",
        np.where(risk_score <= high_thresh, "Medium Risk", "High Risk")
    )

    # ── Assemble DataFrame ────────────────────────────────
    df = pd.DataFrame({
        "student_id":       [f"STU{i:04d}" for i in range(1, n + 1)],
        "gender":           gender,
        "age":              age,
        "residence":        residence,
        "family_income":    family_income,
        "father_education": father_edu,
        "mother_education": mother_edu,
        "department":       department,
        "semester":         semester,
        "scholarship":      scholarship,
        "prev_gpa":         prev_gpa,
        "matric_marks":     matric_marks,
        "inter_marks":      inter_marks,
        "failed_subjects":  failed_subjects,
        "backlogs":         backlogs,
        "attendance_pct":   attendance_pct,
        "study_hours_day":  study_hours_day,
        "library_visits":   library_visits,
        "lms_logins":       lms_logins,
        "assignment_sub_pct": assignment_sub,
        "midterm_score":    midterm_score,
        "stress_level":     stress_level,
        "motivation":       motivation,
        "sleep_hours":      sleep_hours,
        "part_time_job":    part_time_job,
        "commute_hours":    commute_hours,
        "risk_level":       risk_label,
    })

    # Introduce ~3% missing values in realistic columns
    for col in ["study_hours_day", "sleep_hours", "library_visits", "commute_hours"]:
        mask = rng.choice([True, False], n, p=[0.03, 0.97])
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    out_path = out_dir / "students.csv"
    df.to_csv(out_path, index=False)

    print(f"✓ Dataset saved → {out_path}")
    print(f"  Shape       : {df.shape}")
    print(f"  Risk counts :\n{df['risk_level'].value_counts()}")
    print(f"  Missing vals:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
