import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from model import train_model
from evaluator import (
    evaluate_fairness,
    explain_model_ui,
    generate_natural_language_explanation,
    calculate_model_risk_score
)

from report_generator import generate_full_report


# ------------------------------------------------------------
# Page Settings
# ------------------------------------------------------------
st.set_page_config(page_title="AI Overseeing AI", layout="wide")

# Sidebar styling
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #141518;
    padding-top: 30px;
}
.stSelectbox div[data-baseweb="select"] {
    border: 1px solid #4C9AFF !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #4C9AFF;
    margin-top: 10px;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
    margin-top: -10px;
    margin-bottom: 30px;
}
</style>

<h1 class="main-title">Responsible AI Dashboard</h1>
<p class="sub-title">Evaluate fairness • Train ML models • Explain decisions</p>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
action = st.sidebar.selectbox(
    "Select an option",
    [
        "🏠 Home",
        "📊 Train Model (CSV)",
        "🧪 Fairness Evaluation",
        "🔍 Explainability",
        "🚦 Model Risk Score",
        "🧮 Predict on New Data",
        "📦 View Dataset"

    ]
)


# ------------------------------------------------------------
# Card Component
# ------------------------------------------------------------
def card(title, desc, icon="📌", href=None):
    st.markdown(
        f"""
        <div style="
            background-color:#111213;
            border-radius:14px;
            padding:18px 22px;
            margin:16px 0;
            width: 95%;
            box-shadow:0 6px 18px rgba(0,0,0,0.45);
        ">
            <div style="display:flex;align-items:center;gap:16px;">
                <div style="font-size:28px;">{icon}</div>
                <div>
                    <div style="font-weight:700;font-size:18px;color:#fff;">{title}</div>
                    <div style="color:#b8b8b8;font-size:14px;margin-top:4px;">{desc}</div>
                </div>
                <div style="margin-left:auto;color:#9aa7ff;font-size:13px;">{href or ""}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
if action == "🏠 Home":
    st.subheader("Welcome 👋")
    st.write("This dashboard helps you explore models, fairness metrics, explainability, and predictions!")

    col1, col2 = st.columns(2)
    with col1:
        card("Train a Model", "Upload CSV & train quickly.", "📊", "Go to: Train Model")
    with col2:
        card("Fairness Evaluation", "Analyze demographic bias.", "🧪", "Go to: Fairness Evaluation")

    col3, col4 = st.columns(2)
    with col3:
        card("Explainability", "Feature importance & SHAP.", "🔍", "Go to: Explainability")
    with col4:
        card("Prediction Tool", "Predict from new data.", "🧮", "Go to: Predict")


# ------------------------------------------------------------
# TRAIN MODEL PAGE
# ------------------------------------------------------------
elif action == "📊 Train Model (CSV)":
    st.subheader("📊 Train a Machine Learning Model")
    uploaded = st.file_uploader("Upload dataset", type=["csv"])

    if uploaded:
        with open("loan_data.csv", "wb") as f:
            f.write(uploaded.getbuffer())

        st.success("CSV Uploaded Successfully!")

        with st.spinner("Training your model..."):
            try:
                model, X_test, y_test, acc, metrics, cm = train_model("loan_data.csv")

                joblib.dump(model, "trained_model.pkl")

                st.success("🎉 Model trained successfully!")
                st.write(f"### Accuracy: `{metrics['accuracy']:.3f}`")
                st.write(f"Precision: `{metrics['precision']:.3f}`")
                st.write(f"Recall: `{metrics['recall']:.3f}`")
                st.write(f"F1-score: `{metrics['f1']:.3f}`")

                # Confusion Matrix
                fig, ax = plt.subplots()
                ax.matshow(cm, cmap="Blues")
                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        ax.text(j, i, cm[i][j], va='center', ha='center')
                st.pyplot(fig)

                st.success("Model saved as trained_model.pkl")

                # ---- PDF REPORT ----
                st.write("### 📄 Full Report Generator")
                df_tmp = pd.read_csv("loan_data.csv")
                fairness_cols = [
                    c for c in df_tmp.columns if df_tmp[c].dtype == object or df_tmp[c].nunique() <= 10
                ]
                sensitive = st.selectbox("Include fairness attribute (optional)", ["(none)"] + fairness_cols)

                if sensitive == "(none)":
                    sensitive = None

                if st.button("Generate Full PDF Report"):
                    path = generate_full_report(
                        pdf_path="model_report.pdf",
                        model_path="trained_model.pkl",
                        data_path="loan_data.csv",
                        sensitive_attribute=sensitive
                    )

                    with open(path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Report",
                            f,
                            "model_report.pdf",
                            "application/pdf"
                        )

            except Exception as e:
                st.error(f"Training error: {e}")


# ------------------------------------------------------------
# FAIRNESS PAGE
# ------------------------------------------------------------
elif action == "🧪 Fairness Evaluation":
    st.subheader("🧪 Model Fairness Evaluation")
    try:
        evaluate_fairness()
    except Exception as e:
        st.error(f"Fairness evaluation error: {e}")


# ------------------------------------------------------------
# EXPLAINABILITY PAGE (UPDATED WITH NLG)
# ------------------------------------------------------------
elif action == "🔍 Explainability":
    st.subheader("🔍 Model Explainability (SHAP)")
    try:
        result = explain_model_ui()

        # result must return (shap_values, feature_names, row_data)
        if result:
            shap_values, feature_names, row_data = result

            st.markdown("---")
            st.write("### 🗣 Natural Language Explanation")

            explanation = generate_natural_language_explanation(
                shap_values, feature_names, row_data
            )

            st.markdown(explanation)

    except Exception as e:
        st.error(f"Explainability error: {e}")


# ------------------------------------------------------------
# 🚦 MODEL RISK SCORE PAGE
# ------------------------------------------------------------
elif action == "🚦 Model Risk Score":
    st.subheader("🚦 Model Risk Assessment")

    try:
        model = joblib.load("trained_model.pkl")
        df = pd.read_csv("loan_data.csv")
    except:
        st.error("Train a model first!")
        st.stop()

    st.info("Evaluating model risk... ⏳")

    # Preparing SHAP for explainability risk
    import shap
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    X_transformed = preprocessor.transform(X)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)[0]

    # Compute Risk
    final_score, level, components = calculate_model_risk_score(
        model, df, shap_values, X.columns
    )

    # Display
    st.markdown(f"## {level}")
    st.write(f"### Final Risk Score: **{final_score:.3f}**")

    st.write("### 🔍 Risk Breakdown")
    st.json(components)

    # Explanation
    st.write("### 🧠 Interpretation")
    if "LOW" in level:
        st.success("Model is safe and stable. Low bias & strong performance.")
    elif "MEDIUM" in level:
        st.warning("Some bias or instability exists. Review model behavior.")
    else:
        st.error("High risk detected! Model may be unfair or unreliable.")


# ------------------------------------------------------------
# PREDICTION PAGE 
# ------------------------------------------------------------
elif action == "🧮 Predict on New Data":
    st.subheader("🧮 Predict Using Trained Model")

    try:
        model = joblib.load("trained_model.pkl")
        df = pd.read_csv("loan_data.csv")
        target = df.columns[-1]
        features = df.drop(columns=[target]).columns.tolist()

        st.write("### Enter feature values below:")

        user_input = {}
        for col in features:
            if df[col].dtype in ["int64", "float64"]:
                user_input[col] = st.number_input(col, value=float(df[col].median()))
            else:
                user_input[col] = st.selectbox(col, df[col].dropna().unique().tolist())

        if st.button("🔍 Predict"):
            input_df = pd.DataFrame([user_input])
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            st.success(f"### Prediction: **{pred}**")
            st.write(f"### Probability: `{np.max(proba):.3f}`")

    except Exception as e:
        st.error(f"Prediction error: {e}")


# ------------------------------------------------------------
# VIEW DATASET PAGE
# ------------------------------------------------------------
elif action == "📦 View Dataset":
    st.subheader("📦 Dataset Explorer")
    try:
        df = pd.read_csv("loan_data.csv")
        st.dataframe(df)

        st.write("### Dataset Summary")
        st.dataframe(df.describe(include="all"))

        st.write("### Missing Values")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        st.dataframe(missing)

    except Exception as e:
        st.warning(f"Upload dataset first: {e}")
