"""
streamlit_app.py
────────────────────────────────────────────────────────────
Student Performance Predictor — Streamlit Web Application

Run: streamlit run app/streamlit_app.py
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ── Path setup ────────────────────────────────────────────
ROOT = Path(__file__).parent.parent


MODEL_DIR   = ROOT / "models"
DATA_DIR    = ROOT / "data/processed"
FIG_DIR     = ROOT / "reports/figures"

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #ffffff; color: #111827; }
  .block-container { padding: 2rem; }

  /* Sidebar */
  [data-testid="stSidebar"] { background-color: #f3f4f6; }
  [data-testid="stSidebar"] .stMarkdown { color: #374151; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px;
  }

  /* Buttons */
  .stButton > button {
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    font-weight: 600;
    width: 100%;
  }
  .stButton > button:hover { background: #4338ca; }

  /* Risk badge */
  .risk-high   { background:#fee2e2; border:1px solid #ef4444; color:#b91c1c; }
  .risk-medium { background:#fef3c7; border:1px solid #f59e0b; color:#92400e; }
  .risk-low    { background:#dcfce7; border:1px solid #10b981; color:#065f46; }

  .risk-high, .risk-medium, .risk-low {
    border-radius:8px;
    padding:8px 16px;
    font-weight:700;
    text-align:center;
  }

  /* Intervention card */
  .intervention-critical { border-left: 4px solid #ef4444; background:#fee2e2; }
  .intervention-high     { border-left: 4px solid #f59e0b; background:#fef3c7; }
  .intervention-medium   { border-left: 4px solid #6366f1; background:#eef2ff; }
  .intervention-low      { border-left: 4px solid #10b981; background:#dcfce7; }

  .intervention-critical,
  .intervention-high,
  .intervention-medium,
  .intervention-low {
    border-radius:6px;
    padding:10px 14px;
    margin:6px 0;
  }

  h1, h2, h3 { color: #111827 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load(MODEL_DIR / "best_model.joblib")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    le           = joblib.load(MODEL_DIR / "label_encoder.joblib")
    features     = pd.read_csv(DATA_DIR / "feature_names.csv", header=None)[0].tolist()
    return model, preprocessor, le, features


@st.cache_data
def load_summary():
    with open(MODEL_DIR / "training_summary.json") as f:
        return json.load(f)


@st.cache_data
def load_background():
    return np.load(DATA_DIR / "X_train.npy")


LABEL_NAMES  = ["Low Risk", "Medium Risk", "High Risk"]
RISK_COLORS  = {"Low Risk": "#10b981", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"}
RISK_CSS     = {"Low Risk": "risk-low", "Medium Risk": "risk-medium", "High Risk": "risk-high"}


# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮 Predict Risk", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    try:
        summary = load_summary()
        st.markdown(f"**Best Model:** {summary['best_model'].replace('_',' ').title()}")
        st.markdown(f"**Test Accuracy:** {summary['test_results']['accuracy']:.2%}")
        st.markdown(f"**Test F1 Macro:** {summary['test_results']['f1_macro']:.2%}")
    except Exception:
        st.info("Run the pipeline first to load model stats.")


# ─────────────────────────────────────────────────────────
#  PAGE 1 — PREDICT
# ─────────────────────────────────────────────────────────
if "Predict" in page:
    st.title("🔮 Student Risk Prediction")
    st.markdown("Enter student information below to predict academic risk level and receive personalised intervention recommendations.")
    st.markdown("---")

    # ── Input Form ────────────────────────────────────────
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics")
            gender        = st.selectbox("Gender",         ["Male", "Female"])
            age           = st.slider("Age",               16, 30, 20)
            residence     = st.selectbox("Residence",      ["Urban", "Rural", "Suburban"])
            family_income = st.selectbox("Family Income",  ["Low", "Middle", "High"])
            father_edu    = st.selectbox("Father's Education",
                           ["None","Primary","Secondary","Graduate","Postgrad"])
            mother_edu    = st.selectbox("Mother's Education",
                           ["None","Primary","Secondary","Graduate","Postgrad"])
            department    = st.selectbox("Department",
                           ["CS", "SE", "IT", "EE", "ME", "BBA", "Economics"])
            semester      = st.slider("Current Semester", 1, 8, 3)
            scholarship   = st.selectbox("Scholarship",   ["No", "Yes"])

        with col2:
            st.markdown("#### 📚 Academic Record")
            prev_gpa      = st.slider("Previous GPA",       0.0, 4.0, 2.8, 0.1)
            matric_marks  = st.slider("Matric Marks (%)",   33, 100, 72)
            inter_marks   = st.slider("Intermediate (%)",   33, 100, 68)
            failed_subj   = st.slider("Failed Subjects",    0, 6, 0)
            backlogs      = st.slider("Backlogs",           0, 6, 0)
            midterm_score = st.slider("Midterm Score (/100)", 0, 100, 55)
            assign_sub    = st.slider("Assignments Submitted (%)", 0, 100, 80)

        with col3:
            st.markdown("#### 📈 Engagement & Lifestyle")
            attendance    = st.slider("Attendance (%)",     10, 100, 76)
            study_hrs     = st.slider("Study Hours/Day",    0.0, 12.0, 3.0, 0.5)
            lib_visits    = st.slider("Library Visits/Month", 0, 30, 5)
            lms_logins    = st.slider("LMS Logins/Month",   0, 60, 15)
            stress        = st.slider("Stress Level (1-10)",1, 10, 5)
            motivation    = st.slider("Motivation (1-10)",  1, 10, 6)
            sleep_hrs     = st.slider("Sleep Hours/Night",  3.0, 10.0, 6.5, 0.5)
            part_time     = st.selectbox("Part-Time Job",   ["No", "Yes"])
            commute       = st.slider("Commute Hours/Day",  0.0, 5.0, 1.0, 0.5)

        submitted = st.form_submit_button("🔮 Predict Risk Level", use_container_width=True)

    # ── Prediction ────────────────────────────────────────
    if submitted:
        try:
            model, preprocessor, le, features = load_model()
        except Exception as e:
            st.error(f"Model not found. Run the pipeline first: `python run_pipeline.py`\n\n{e}")
            st.stop()

        # Build raw input dict
        raw = {
            "gender":           gender,
            "age":              age,
            "residence":        residence,
            "family_income":    family_income,
            "father_education": father_edu,
            "mother_education": mother_edu,
            "department":       department,
            "semester":         semester,
            "scholarship":      1 if scholarship == "Yes" else 0,
            "prev_gpa":         prev_gpa,
            "matric_marks":     matric_marks,
            "inter_marks":      inter_marks,
            "failed_subjects":  failed_subj,
            "backlogs":         backlogs,
            "attendance_pct":   attendance,
            "study_hours_day":  study_hrs,
            "library_visits":   lib_visits,
            "lms_logins":       lms_logins,
            "assignment_sub_pct": assign_sub,
            "midterm_score":    midterm_score,
            "stress_level":     stress,
            "motivation":       motivation,
            "sleep_hours":      sleep_hrs,
            "part_time_job":    1 if part_time == "Yes" else 0,
            "commute_hours":    commute,
        }

        # Apply feature engineering
        from src.features.feature_engineering import engineer_features
        input_df = engineer_features(pd.DataFrame([raw]))

        # Preprocess
        X_proc  = preprocessor.transform(input_df)
        pred    = model.predict(X_proc)[0]
        proba   = model.predict_proba(X_proc)[0]
        label   = LABEL_NAMES[pred]

        st.markdown("---")
        st.markdown("## 📋 Prediction Results")

        # ── Risk badge + probabilities ─────────────────────
        r1, r2, r3, r4 = st.columns([1.5, 1, 1, 1])
        with r1:
            st.markdown(f'<div class="{RISK_CSS[label]}">🎯 {label}</div>',
                        unsafe_allow_html=True)
        with r2:
            st.metric("🟢 Low Risk",    f"{proba[0]:.1%}")
        with r3:
            st.metric("🟡 Medium Risk", f"{proba[1]:.1%}")
        with r4:
            st.metric("🔴 High Risk",   f"{proba[2]:.1%}")

        # ── Probability bar ────────────────────────────────
        st.markdown("#### Confidence Distribution")
        prob_df = pd.DataFrame({"Risk Level": LABEL_NAMES, "Probability": proba})
        st.bar_chart(prob_df.set_index("Risk Level"), color=["#6366f1"])

        # ── Interventions ──────────────────────────────────
        st.markdown("---")
        st.markdown("## 🎯 Intervention Recommendations")

        from src.models.interventions import get_interventions
        interventions = get_interventions(raw, label)

        if not interventions:
            st.success("No significant risk factors detected. Student is performing well.")
        else:
            for inv in interventions:
                css_class = f"intervention-{inv.priority.lower()}"
                st.markdown(f"""
<div class="{css_class}">
  <b>[{inv.priority}] {inv.category}</b><br>
  {inv.action}<br>
  <small>👤 <b>Responsible:</b> {inv.responsible} &nbsp;|&nbsp; ⏱ <b>Timeline:</b> {inv.timeline}</small>
</div>""", unsafe_allow_html=True)

        # ── Key risk factors ───────────────────────────────
        st.markdown("---")
        st.markdown("## 🔍 Key Risk Indicators")
        fa1, fa2 = st.columns(2)

        risk_factors = []
        if attendance < 65:
            risk_factors.append(f"⚠ Low attendance: {attendance:.0f}%")
        if midterm_score < 50:
            risk_factors.append(f"⚠ Low midterm score: {midterm_score:.0f}/100")
        if prev_gpa < 2.0:
            risk_factors.append(f"⚠ Low GPA: {prev_gpa:.2f}")
        if stress >= 7:
            risk_factors.append(f"⚠ High stress: {stress}/10")
        if failed_subj >= 2:
            risk_factors.append(f"⚠ Failed subjects: {failed_subj}")
        if motivation <= 3:
            risk_factors.append(f"⚠ Low motivation: {motivation}/10")
        if study_hrs < 1.5:
            risk_factors.append(f"⚠ Low study hours: {study_hrs:.1f}h/day")

        protectors = []
        if attendance >= 85:
            protectors.append(f"✓ Strong attendance: {attendance:.0f}%")
        if motivation >= 7:
            protectors.append(f"✓ High motivation: {motivation}/10")
        if prev_gpa >= 3.0:
            protectors.append(f"✓ Good GPA: {prev_gpa:.2f}")
        if scholarship == "Yes":
            protectors.append("✓ Has scholarship support")
        if study_hrs >= 4:
            protectors.append(f"✓ Dedicated study: {study_hrs:.1f}h/day")

        with fa1:
            st.markdown("**Risk Factors**")
            for f in risk_factors or ["None identified"]:
                st.markdown(f)
        with fa2:
            st.markdown("**Protective Factors**")
            for p in protectors or ["Build more strengths"]:
                st.markdown(p)


# ─────────────────────────────────────────────────────────
#  PAGE 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────
elif "Performance" in page:
    st.title("📊 Model Performance")

    try:
        summary = load_summary()
    except Exception:
        st.warning("Run `python run_pipeline.py` to generate model artifacts.")
        st.stop()

    # ── Test metrics ───────────────────────────────────────
    st.markdown("### 🏆 Best Model — Test Set Results")
    st.markdown(f"**Model:** `{summary['best_model'].replace('_', ' ').title()}`")

    m = summary["test_results"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{m['accuracy']:.2%}")
    c2.metric("F1 Macro",  f"{m['f1_macro']:.2%}")
    c3.metric("Precision", f"{m['precision']:.2%}")
    c4.metric("Recall",    f"{m['recall']:.2%}")

    # ── Model comparison table ─────────────────────────────
    st.markdown("### 📈 Model Comparison (Validation Set)")
    val = summary.get("val_results", {})
    if val:
        rows = []
        for model_name, metrics in val.items():
            rows.append({
                "Model":     model_name.replace("_", " ").title(),
                "Accuracy":  f"{metrics['accuracy']:.4f}",
                "F1 Macro":  f"{metrics['f1_macro']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall":    f"{metrics['recall']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Figures ────────────────────────────────────────────
    st.markdown("### 📷 Evaluation Plots")

    fig_map = {
        "model_comparison.png":    "Model Comparison",
        "confusion_matrix.png":    "Confusion Matrix",
        "roc_curves.png":          "ROC Curves",
        "precision_recall.png":    "Precision-Recall Curves",
        "feature_importance.png":  "Feature Importance",
        "shap_bar_global.png":     "SHAP Global Importance",
        "shap_summary_class2.png": "SHAP — High Risk Class",
    }

    for fname, label in fig_map.items():
        fpath = FIG_DIR / fname
        if fpath.exists():
            st.markdown(f"#### {label}")
            st.image(str(fpath), use_column_width=True)


# ─────────────────────────────────────────────────────────
#  PAGE 3 — ABOUT
# ─────────────────────────────────────────────────────────
elif "About" in page:
    st.title("ℹ️ About This Project")
    st.markdown("""
### AI-Based Student Performance Predictor

This system uses machine learning to predict student academic risk levels
and generate targeted intervention recommendations — helping academic advisors
take timely, evidence-based action.

---

#### 🔬 ML Pipeline
| Stage | Details |
|-------|---------|
| **Dataset** | 2,000 synthetic students with 26 features |
| **Feature Engineering** | 10 domain-driven derived features |
| **Models Trained** | Logistic Regression, Random Forest, XGBoost |
| **Tuning** | RandomizedSearchCV with 5-fold StratifiedKFold |
| **Evaluation** | Accuracy, F1 (macro), Precision, Recall, ROC-AUC |
| **Explainability** | SHAP values (TreeExplainer) |

#### 🎯 Risk Levels
| Level | Description |
|-------|-------------|
| 🟢 Low Risk | On track — minimal intervention needed |
| 🟡 Medium Risk | Early warning — proactive support recommended |
| 🔴 High Risk | Immediate intervention required |

#### 🛠️ Tech Stack
`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `SHAP` · `Streamlit` · `Matplotlib` · `Seaborn`

---

**Author:** Khadija Ayub | BSCS @ NUML Islamabad
**GitHub:** [github.com/khadija-ayub](https://github.com/khadija-ayub)
    """)
