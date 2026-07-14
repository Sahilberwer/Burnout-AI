"""
app.py
================================================================
Prediction of Burnout Risk Level Using Decision Tree & Random Forest
Honeywell AI Internship Project

Main Streamlit application entry point.
Run with: streamlit run app.py
================================================================
"""

import time

import pandas as pd
import streamlit as st

from predict import ModelLoadError, predict_burnout_risk
from preprocessing import CATEGORICAL_FIELD_META, FEATURE_COLUMNS, NUMERIC_FIELD_META
from utils import (
    artifact_image_path,
    badge_row,
    get_dataset,
    get_feature_importance,
    get_metrics,
    get_model_and_encoders,
    inject_custom_css,
    render_footer,
)

# ----------------------------------------------------------------
# Page config (must be first Streamlit call)
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Burnout Risk AI | Student Wellness Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.6rem 0 1rem 0;">
            <div style="font-size:2.4rem;">🧠</div>
            <div style="font-weight:800; font-size:1.15rem; color:#F1F5F9;">Burnout Risk AI</div>
            <div style="font-size:0.78rem; color:#A1A1AA;">Student Wellness Predictor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🎯 Prediction", "📊 Model Performance", "📈 Visualizations", "👤 About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.78rem; color:#A1A1AA; line-height:1.6;">
        <b>Final Model:</b> Random Forest<br>
        <b>Dataset:</b> AI Student Impact (50,000 rows)<br>
        <b>Target:</b> Burnout Risk Level<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    dark_mode = st.toggle("🌙 Dark Mode", value=True, help="Toggle between dark and light themes")

if not dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf7 100%) !important; }
        h1, h2, h3, h4, p, span, label, li, div { color: #1E293B !important; }
        .glass-card { background: rgba(255,255,255,0.7) !important; border: 1px solid rgba(0,0,0,0.08) !important; }
        .hero-banner p, .hero-banner h1 { color: white !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# PAGE: HOME
# ================================================================
def render_home():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🧠 Prediction of Burnout Risk Level</h1>
            <p>Using Decision Tree &amp; Random Forest to detect student burnout risk from academic performance,
            GenAI usage patterns, and study habits.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    badge_row(["🐍 Python", "🌲 Random Forest", "🌳 Decision Tree", "🎨 Streamlit",
               "📊 Scikit-learn", "🏢 Honeywell AI Internship"])

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    metrics = get_metrics()
    rf_acc = metrics.get("random_forest", {}).get("accuracy")
    with col1:
        st.markdown(
            f"""<div class="glass-card">
            <h3>🎯 Objective</h3>
            <p>Predict a student's Burnout Risk Level (Low / Medium / High) to enable early,
            personalized wellness interventions.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        acc_str = f"{rf_acc*100:.1f}%" if rf_acc else "—"
        st.markdown(
            f"""<div class="glass-card">
            <h3>📊 Dataset</h3>
            <p>50,000 student records covering GPA, GenAI usage, study habits, and exam anxiety.
            Test accuracy: <b>{acc_str}</b></p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class="glass-card">
            <h3>🌲 Final Model</h3>
            <p>Random Forest Classifier, selected after comparing against a tuned Decision Tree
            using GridSearchCV hyperparameter optimization.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### 🔄 Project Workflow")
    steps = [
        ("1️⃣ Data Cleaning", "Drop identifiers, handle missing values"),
        ("2️⃣ Encoding", "Label-encode categorical features"),
        ("3️⃣ EDA", "Correlation heatmap, class distribution, feature analysis"),
        ("4️⃣ Modeling", "Train Decision Tree & Random Forest"),
        ("5️⃣ Tuning", "GridSearchCV hyperparameter optimization"),
        ("6️⃣ Deployment", "Interactive Streamlit prediction app"),
    ]
    cols = st.columns(len(steps))
    for c, (title, desc) in zip(cols, steps):
        with c:
            st.markdown(
                f"""<div class="glass-card" style="text-align:center; min-height:130px;">
                <div style="font-weight:700;">{title}</div>
                <div style="font-size:0.82rem; color:#A1A1AA; margin-top:0.4rem;">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### ✨ Project Highlights")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown(
            """<div class="glass-card">
            <ul>
            <li>Real dataset of 50,000 student records — no synthetic placeholders</li>
            <li>Two models compared: Decision Tree vs Random Forest</li>
            <li>Hyperparameter tuning via GridSearchCV with cross-validation</li>
            <li>Full evaluation suite: accuracy, precision, recall, F1, ROC-AUC</li>
            </ul>
            </div>""",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            """<div class="glass-card">
            <ul>
            <li>Auto-generated prediction form matching dataset schema</li>
            <li>Confidence scores and per-class probability breakdown</li>
            <li>Actionable, risk-level-specific recommendations</li>
            <li>Deployable as-is to Streamlit Cloud, Render, or HuggingFace Spaces</li>
            </ul>
            </div>""",
            unsafe_allow_html=True,
        )

    st.info("🏢 Developed as part of the **Honeywell AI Internship** — demonstrating an end-to-end "
            "ML workflow from raw data to a deployed, production-ready web application.")


# ================================================================
# PAGE: PREDICTION
# ================================================================
def render_prediction():
    st.markdown(
        """<div class="hero-banner"><h1>🎯 Burnout Risk Prediction</h1>
        <p>Fill in the student profile below to get an instant AI-powered burnout risk assessment.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    try:
        model, encoders, _ = get_model_and_encoders()
    except ModelLoadError as e:
        st.error(f"⚠️ {e}")
        return

    with st.form("prediction_form"):
        st.markdown("#### 📋 Student Profile")
        c1, c2 = st.columns(2)

        raw_input = {}
        with c1:
            st.markdown("**Academic & Demographic**")
            raw_input["Major_Category"] = st.selectbox(
                CATEGORICAL_FIELD_META["Major_Category"]["label"],
                CATEGORICAL_FIELD_META["Major_Category"]["options"],
            )
            raw_input["Year_of_Study"] = st.selectbox(
                CATEGORICAL_FIELD_META["Year_of_Study"]["label"],
                CATEGORICAL_FIELD_META["Year_of_Study"]["options"],
            )
            raw_input["Pre_Semester_GPA"] = st.slider(
                NUMERIC_FIELD_META["Pre_Semester_GPA"]["label"],
                min_value=NUMERIC_FIELD_META["Pre_Semester_GPA"]["min"],
                max_value=NUMERIC_FIELD_META["Pre_Semester_GPA"]["max"],
                value=NUMERIC_FIELD_META["Pre_Semester_GPA"]["mean"],
                step=NUMERIC_FIELD_META["Pre_Semester_GPA"]["step"],
                help=NUMERIC_FIELD_META["Pre_Semester_GPA"]["help"],
            )
            raw_input["Post_Semester_GPA"] = st.slider(
                NUMERIC_FIELD_META["Post_Semester_GPA"]["label"],
                min_value=NUMERIC_FIELD_META["Post_Semester_GPA"]["min"],
                max_value=NUMERIC_FIELD_META["Post_Semester_GPA"]["max"],
                value=NUMERIC_FIELD_META["Post_Semester_GPA"]["mean"],
                step=NUMERIC_FIELD_META["Post_Semester_GPA"]["step"],
                help=NUMERIC_FIELD_META["Post_Semester_GPA"]["help"],
            )
            raw_input["Traditional_Study_Hours"] = st.number_input(
                NUMERIC_FIELD_META["Traditional_Study_Hours"]["label"],
                min_value=NUMERIC_FIELD_META["Traditional_Study_Hours"]["min"],
                max_value=NUMERIC_FIELD_META["Traditional_Study_Hours"]["max"],
                value=NUMERIC_FIELD_META["Traditional_Study_Hours"]["mean"],
                step=NUMERIC_FIELD_META["Traditional_Study_Hours"]["step"],
                help=NUMERIC_FIELD_META["Traditional_Study_Hours"]["help"],
            )
            raw_input["Anxiety_Level_During_Exams"] = st.slider(
                NUMERIC_FIELD_META["Anxiety_Level_During_Exams"]["label"],
                min_value=int(NUMERIC_FIELD_META["Anxiety_Level_During_Exams"]["min"]),
                max_value=int(NUMERIC_FIELD_META["Anxiety_Level_During_Exams"]["max"]),
                value=int(NUMERIC_FIELD_META["Anxiety_Level_During_Exams"]["mean"]),
                help=NUMERIC_FIELD_META["Anxiety_Level_During_Exams"]["help"],
            )
            raw_input["Skill_Retention_Score"] = st.slider(
                NUMERIC_FIELD_META["Skill_Retention_Score"]["label"],
                min_value=NUMERIC_FIELD_META["Skill_Retention_Score"]["min"],
                max_value=NUMERIC_FIELD_META["Skill_Retention_Score"]["max"],
                value=NUMERIC_FIELD_META["Skill_Retention_Score"]["mean"],
                step=NUMERIC_FIELD_META["Skill_Retention_Score"]["step"],
                help=NUMERIC_FIELD_META["Skill_Retention_Score"]["help"],
            )

        with c2:
            st.markdown("**GenAI Usage Patterns**")
            raw_input["Primary_Use_Case"] = st.selectbox(
                CATEGORICAL_FIELD_META["Primary_Use_Case"]["label"],
                CATEGORICAL_FIELD_META["Primary_Use_Case"]["options"],
            )
            raw_input["Prompt_Engineering_Skill"] = st.radio(
                CATEGORICAL_FIELD_META["Prompt_Engineering_Skill"]["label"],
                CATEGORICAL_FIELD_META["Prompt_Engineering_Skill"]["options"],
                horizontal=True,
            )
            raw_input["Weekly_GenAI_Hours"] = st.slider(
                NUMERIC_FIELD_META["Weekly_GenAI_Hours"]["label"],
                min_value=NUMERIC_FIELD_META["Weekly_GenAI_Hours"]["min"],
                max_value=NUMERIC_FIELD_META["Weekly_GenAI_Hours"]["max"],
                value=NUMERIC_FIELD_META["Weekly_GenAI_Hours"]["mean"],
                step=NUMERIC_FIELD_META["Weekly_GenAI_Hours"]["step"],
                help=NUMERIC_FIELD_META["Weekly_GenAI_Hours"]["help"],
            )
            raw_input["Tool_Diversity"] = st.slider(
                NUMERIC_FIELD_META["Tool_Diversity"]["label"],
                min_value=int(NUMERIC_FIELD_META["Tool_Diversity"]["min"]),
                max_value=int(NUMERIC_FIELD_META["Tool_Diversity"]["max"]),
                value=int(NUMERIC_FIELD_META["Tool_Diversity"]["mean"]),
                help=NUMERIC_FIELD_META["Tool_Diversity"]["help"],
            )
            raw_input["Perceived_AI_Dependency"] = st.slider(
                NUMERIC_FIELD_META["Perceived_AI_Dependency"]["label"],
                min_value=int(NUMERIC_FIELD_META["Perceived_AI_Dependency"]["min"]),
                max_value=int(NUMERIC_FIELD_META["Perceived_AI_Dependency"]["max"]),
                value=int(NUMERIC_FIELD_META["Perceived_AI_Dependency"]["mean"]),
                help=NUMERIC_FIELD_META["Perceived_AI_Dependency"]["help"],
            )
            paid_sub = st.checkbox("💳 Has Paid AI Subscription?", value=False)
            raw_input["Paid_Subscription"] = "True" if paid_sub else "False"
            raw_input["Institutional_Policy"] = st.selectbox(
                CATEGORICAL_FIELD_META["Institutional_Policy"]["label"],
                CATEGORICAL_FIELD_META["Institutional_Policy"]["options"],
            )

        submitted = st.form_submit_button("🔮 Predict Burnout Risk", use_container_width=True)

    if submitted:
        # sanity check all fields present
        missing = [c for c in FEATURE_COLUMNS if c not in raw_input or raw_input[c] is None]
        if missing:
            st.error(f"⚠️ Missing input(s): {', '.join(missing)}. Please fill in every field.")
            return

        with st.spinner("Analyzing student profile..."):
            time.sleep(0.4)
            try:
                result = predict_burnout_risk(raw_input, model=model, encoders=encoders)
            except ValueError as e:
                st.error(f"⚠️ {e}")
                return

        st.session_state.history.append({**raw_input, "Prediction": result["prediction"],
                                          "Confidence": f"{result['confidence']*100:.1f}%"})

        st.markdown(
            f"""<div class="risk-banner" style="background:{result['color']}22; border:2px solid {result['color']};">
            <span style="color:{result['color']};">Predicted Burnout Risk: {result['prediction'].upper()}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([1, 1.4])
        with col_a:
            st.metric("Confidence Score", f"{result['confidence']*100:.1f}%")
            st.progress(result["confidence"])
            st.markdown("**Probability Breakdown**")
            for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
                st.write(f"{label}: {prob*100:.1f}%")
                st.progress(prob)

        with col_b:
            st.markdown("**💡 Explanation & Recommendations**")
            st.markdown(
                f"""<div class="glass-card">
                Based on this profile — GenAI usage, study habits, and exam anxiety patterns —
                the model estimates a <b>{result['prediction']}</b> burnout risk with
                {result['confidence']*100:.1f}% confidence.
                </div>""",
                unsafe_allow_html=True,
            )
            for rec in result["recommendations"]:
                st.markdown(f"- {rec}")

        st.success("✅ Prediction complete!")

    if st.session_state.history:
        st.markdown("### 🕒 Prediction History (this session)")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        csv = hist_df.to_csv(index=False).encode("utf-8")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.download_button("⬇️ Download History as CSV", csv, "prediction_history.csv", "text/csv",
                                use_container_width=True)
        with c2:
            if st.button("🔄 Reset History", use_container_width=True):
                st.session_state.history = []
                st.rerun()


# ================================================================
# PAGE: MODEL PERFORMANCE
# ================================================================
def render_model_performance():
    st.markdown(
        """<div class="hero-banner"><h1>📊 Model Performance</h1>
        <p>Decision Tree vs Random Forest — full evaluation metrics after GridSearchCV tuning.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    metrics = get_metrics()
    if not metrics:
        st.warning("⚠️ Metrics not found. Please run `python train_model.py` first.")
        return

    rf = metrics["random_forest"]
    dt = metrics["decision_tree"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RF Accuracy", f"{rf['accuracy']*100:.2f}%")
    m2.metric("RF Precision", f"{rf['precision']*100:.2f}%")
    m3.metric("RF Recall", f"{rf['recall']*100:.2f}%")
    m4.metric("RF F1 Score", f"{rf['f1_score']*100:.2f}%")

    st.markdown("#### 🌲 vs 🌳 Model Comparison")
    comp_df = pd.DataFrame(metrics["comparison_table"])
    st.dataframe(comp_df.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}",
                                        "Recall": "{:.4f}", "F1 Score": "{:.4f}"}),
                 use_container_width=True)
    st.image(artifact_image_path("model_comparison.png"), use_container_width=True)

    st.markdown("#### ⚙️ Best Hyperparameters (GridSearchCV)")
    hp1, hp2 = st.columns(2)
    with hp1:
        st.markdown("**Decision Tree**")
        st.json(metrics["best_dt_params"])
    with hp2:
        st.markdown("**Random Forest**")
        st.json(metrics["best_rf_params"])

    st.markdown("#### 🧾 Confusion Matrices")
    cm1, cm2 = st.columns(2)
    with cm1:
        st.image(artifact_image_path("confusion_matrix_dt.png"), caption="Decision Tree", use_container_width=True)
    with cm2:
        st.image(artifact_image_path("confusion_matrix_rf.png"), caption="Random Forest", use_container_width=True)

    st.markdown("#### 📈 ROC Curve (Random Forest, One-vs-Rest)")
    st.image(artifact_image_path("roc_curve.png"), use_container_width=True)

    if rf.get("cross_val_accuracy_mean"):
        st.info(f"🔁 Random Forest 5-fold Cross-Validation Accuracy: "
                f"**{rf['cross_val_accuracy_mean']*100:.2f}% (± {rf['cross_val_accuracy_std']*100:.2f}%)**")

    with st.expander("📄 Full Classification Report — Random Forest"):
        st.dataframe(pd.DataFrame(rf["classification_report"]).T, use_container_width=True)
    with st.expander("📄 Full Classification Report — Decision Tree"):
        st.dataframe(pd.DataFrame(dt["classification_report"]).T, use_container_width=True)


# ================================================================
# PAGE: VISUALIZATIONS
# ================================================================
def render_visualizations():
    st.markdown(
        """<div class="hero-banner"><h1>📈 Data Visualizations</h1>
        <p>Exploratory analysis of the AI Student Impact dataset.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    df = get_dataset()
    importance = get_feature_importance()

    st.markdown("#### 🔥 Correlation Heatmap")
    st.image(artifact_image_path("correlation_heatmap.png"), use_container_width=True)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("#### 🎯 Burnout Risk Class Distribution")
        st.image(artifact_image_path("class_distribution.png"), use_container_width=True)
    with v2:
        st.markdown("#### 🌟 Feature Importance (Random Forest)")
        st.image(artifact_image_path("feature_importance.png"), use_container_width=True)

    if importance:
        st.markdown("#### 📋 Feature Importance Table")
        imp_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
        imp_df = imp_df.sort_values("Importance", ascending=False)
        st.dataframe(imp_df, use_container_width=True)

    st.markdown("#### 🔎 Explore the Raw Dataset")
    st.caption(f"Showing a sample of the {len(df):,}-row cleaned dataset.")
    st.dataframe(df.sample(min(200, len(df)), random_state=42), use_container_width=True)


# ================================================================
# PAGE: ABOUT
# ================================================================
def render_about():
    st.markdown(
        """<div class="hero-banner"><h1>👤 About This Project</h1>
        <p>Developer, technology stack, and project background.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns([1, 1.6])
    with a1:
        st.markdown(
            """<div class="glass-card" style="text-align:center;">
            <div style="font-size:3rem;">🧑‍💻</div>
            <h3>Sandeep</h3>
            <p style="color:#A1A1AA;">B.Tech CSE (AI &amp; ML), GJU&amp;ST Hisar</p>
            <a href="https://github.com/Sahilberwer" target="_blank">🔗 GitHub</a><br>
            <a href="https://linkedin.com/in/sandeep-berwer" target="_blank">🔗 LinkedIn</a>
            </div>""",
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """<div class="glass-card">
            <h3>🏢 Project Background</h3>
            <p>This project was developed as part of a <b>Honeywell AI Internship</b>, applying a
            complete supervised machine learning workflow to predict student burnout risk from
            academic performance and Generative AI usage patterns.</p>
            <p>It demonstrates end-to-end ML engineering: data cleaning, encoding, exploratory
            analysis, model training and tuning, evaluation, and deployment as an interactive
            production-ready web application.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### 🛠️ Technology Stack")
    badge_row(["Python 3.11+", "Pandas", "NumPy", "Scikit-learn", "Streamlit",
               "Matplotlib", "Seaborn", "GridSearchCV"])

    st.markdown("#### 📁 Repository Structure")
    st.code(
        """burnout-risk-prediction/
├── app.py                  # Streamlit application
├── train_model.py          # Full training pipeline
├── predict.py               # Inference logic
├── preprocessing.py         # Shared preprocessing utilities
├── utils.py                 # Streamlit helpers (CSS, caching)
├── requirements.txt
├── runtime.txt
├── Procfile
├── artifacts/                # model.pkl, encoder.pkl, metrics, charts
├── assets/                   # logo, banner, background images
└── README.md""",
        language="text",
    )

    st.markdown("#### 🔮 Future Scope")
    st.markdown(
        """<div class="glass-card">
        <ul>
        <li>Incorporate longitudinal (multi-semester) data to track burnout trends over time</li>
        <li>Add SHAP-based explainability for individual predictions</li>
        <li>Integrate with institutional LMS platforms for automated early-warning alerts</li>
        <li>Experiment with gradient boosting models (XGBoost / LightGBM) for further accuracy gains</li>
        </ul>
        </div>""",
        unsafe_allow_html=True,
    )


# ================================================================
# ROUTER
# ================================================================
PAGES = {
    "🏠 Home": render_home,
    "🎯 Prediction": render_prediction,
    "📊 Model Performance": render_model_performance,
    "📈 Visualizations": render_visualizations,
    "👤 About": render_about,
}

PAGES[page]()
render_footer()
