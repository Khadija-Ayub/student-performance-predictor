"""
test_pipeline.py
────────────────────────────────────────────────────────────
Unit tests for core pipeline components.
Run: pytest tests/ -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.generate_dataset       import generate_dataset
from src.features.feature_engineering import engineer_features
from src.models.interventions         import get_interventions, Intervention


# ── Dataset generation tests ──────────────────────────────
class TestDatasetGeneration:

    def test_shape(self):
        df = generate_dataset(n=100)
        assert df.shape[0] == 100
        assert df.shape[1] >= 20

    def test_target_classes(self):
        df = generate_dataset(n=500)
        classes = set(df["risk_level"].unique())
        assert classes == {"Low Risk", "Medium Risk", "High Risk"}

    def test_no_all_nulls(self):
        df = generate_dataset(n=200)
        # No column should be entirely null
        assert not any(df.isnull().all())

    def test_gpa_range(self):
        df = generate_dataset(n=300)
        assert df["prev_gpa"].between(0, 4).all()

    def test_attendance_range(self):
        df = generate_dataset(n=300)
        assert df["attendance_pct"].between(0, 100).all()

    def test_student_ids_unique(self):
        df = generate_dataset(n=200)
        assert df["student_id"].nunique() == 200


# ── Feature engineering tests ────────────────────────────
class TestFeatureEngineering:

    @pytest.fixture
    def sample_df(self):
        df = generate_dataset(n=100)
        return df

    def test_new_columns_added(self, sample_df):
        eng = engineer_features(sample_df)
        expected = [
            "academic_momentum", "engagement_score", "study_efficiency",
            "support_index", "stress_motivation_ratio", "lifestyle_score",
            "academic_burden", "gpa_band", "attendance_category", "is_early_semester"
        ]
        for col in expected:
            assert col in eng.columns, f"Missing: {col}"

    def test_academic_momentum_range(self, sample_df):
        eng = engineer_features(sample_df)
        assert eng["academic_momentum"].between(0, 100).all()

    def test_engagement_score_range(self, sample_df):
        eng = engineer_features(sample_df)
        assert eng["engagement_score"].between(0, 100).all()

    def test_no_new_nulls_in_derived(self, sample_df):
        eng = engineer_features(sample_df)
        derived = ["academic_momentum", "engagement_score", "support_index"]
        for col in derived:
            assert eng[col].isnull().sum() == 0, f"Nulls in {col}"


# ── Intervention engine tests ────────────────────────────
class TestInterventions:

    def test_high_risk_generates_interventions(self):
        student = {
            "attendance_pct":    40,
            "midterm_score":     25,
            "prev_gpa":          1.2,
            "failed_subjects":   3,
            "study_hours_day":   0.5,
            "stress_level":      9,
            "motivation":        2,
            "family_income":     "Low",
            "scholarship":       0,
            "part_time_job":     1,
            "assignment_sub_pct": 30,
            "sleep_hours":       5,
        }
        invs = get_interventions(student, "High Risk")
        assert len(invs) > 0
        priorities = [i.priority for i in invs]
        assert "Critical" in priorities or "High" in priorities

    def test_low_risk_gets_enrichment(self):
        student = {
            "attendance_pct":    95,
            "midterm_score":     88,
            "prev_gpa":          3.8,
            "failed_subjects":   0,
            "study_hours_day":   5,
            "stress_level":      3,
            "motivation":        9,
            "family_income":     "High",
            "scholarship":       1,
            "part_time_job":     0,
            "assignment_sub_pct": 98,
            "sleep_hours":       7.5,
        }
        invs = get_interventions(student, "Low Risk")
        # Should get enrichment suggestion
        assert len(invs) >= 1
        assert any(i.category == "Enrichment" for i in invs)

    def test_interventions_sorted_by_priority(self):
        student = {
            "attendance_pct":    45,
            "midterm_score":     30,
            "prev_gpa":          1.0,
            "failed_subjects":   4,
            "study_hours_day":   0.5,
            "stress_level":      9,
            "motivation":        2,
            "family_income":     "Low",
            "scholarship":       0,
            "part_time_job":     1,
            "assignment_sub_pct": 20,
            "sleep_hours":       4,
        }
        invs = get_interventions(student, "High Risk")
        priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        ranks = [priority_order[i.priority] for i in invs]
        assert ranks == sorted(ranks), "Interventions not sorted by priority"

    def test_intervention_has_all_fields(self):
        student = {"attendance_pct": 40, "midterm_score": 30, "prev_gpa": 1.5,
                   "failed_subjects": 2, "study_hours_day": 1.0, "stress_level": 8,
                   "motivation": 3, "family_income": "Low", "scholarship": 0,
                   "part_time_job": 0, "assignment_sub_pct": 40, "sleep_hours": 5}
        invs = get_interventions(student, "High Risk")
        for inv in invs:
            assert inv.category
            assert inv.priority
            assert inv.action
            assert inv.responsible
            assert inv.timeline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
