# ===============================
# app.py — Clean Professional + Futuristic UI (Blue→Purple Gradient)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import your modules
from model import train_model
from evaluator import (
    evaluate_fairness,
    explain_model_ui,
    generate_natural_language_explanation,
)
from report_generator import generate_full_report


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Overseeing AI", layout="wide")

# ===============================
# CUSTOM UI THEME (Blue→Purple Gradient Clean Style)
# ===============================
st.markdown(
    """
    <style>

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0f24, #1d093f);
        color: #e8e8f0 !important;
    }

    /* Header Title */
    .main-title {
        text-align:center;
        font-size:48px;
        font-weight:800;
        background: linear-gradient(90deg, #4286ff, #b46bff);
        -webkit-background-clip: text;
        color: transparent;
        margin-top: 5px;
        margin-bottom: 0px;
    }

    .sub-title {
        text-align:center;
        color:#c8c8d8;
        font-size:18px;
        margin-top:-12px;
        margin-bottom:30px;
    }

    /* Clean glass cards */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 20px;
        backdrop-filter: blur(9px);
        border: 1px solid rgba(255,255,255,0.12);
        transition: 0.25s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(147, 105, 255, 0.55);
        box-shadow: 0 4px 26px rgba(92, 120, 255, 0.25);
    }

    .card-title {
        font-size:20px;
        font-weight:700;
        color:#ffffff;
        margin-bottom:4px;
    }

    .card-desc {
        color:#c9c9d9;
        font-size:14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ===============================
# SIDEBAR NAVIGATION
# ===============================
action = st.sidebar.selectbox(
    "Navigation",
    [
        "🏠 Home",
        "📊 Train Model (CSV)",
        "🧪 Fairness Evaluation",
        "🔍 Explainability",
        "📈 Feature Insights",
        "🚦 Model Risk Score",
        "🧮 Predict on New Data",
        "📦 View Dataset",
    ]
)

# Helper loaders
def safe_load_data():
    if os.path.exists("loan_data.csv"):
        return pd.read_csv("loan_data.csv")
    return None

def safe_load_model():
    if os.path.exists("trained_model.pkl"):
        return joblib.load("trained_model.pkl")
    return None


# ===============================
# HOME PAGE
# ===============================
if action == "🏠 Home":

    st.markdown("<h1 class='main-title'>Responsible AI Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Train models • Check fairness • Explain AI decisions</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:30px;">📊</div>
                <div class="card-title">Train a Model</div>
                <div class="card-desc">Upload your dataset and train ML models instantly.</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:30px;">🧪</div>
                <div class="card-title">Fairness Evaluation</div>
                <div class="card-desc">Check for demographic bias across sensitive groups.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:30px;">🔍</div>
                <div class="card-title">Explainability</div>
                <div class="card-desc">Use SHAP to understand why the model made a decision.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:30px;">🧮</div>
                <div class="card-title">Prediction Tool</div>
                <div class="card-desc">Predict outcomes from user-filled values.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ===============================
# TRAIN MODEL PAGE
# ===============================
elif action == "📊 Train Model (CSV)":
    st.subheader("📊 Train a Machine Learning Model")

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded:
        with open("loan_data.csv", "wb") as f:
            f.write(uploaded.getbuffer())

        st.success("Dataset uploaded!")

        with st.spinner("Training model..."):
            try:
                model, X_test, y_test, acc, metrics, cm = train_model("loan_data.csv")
                joblib.dump(model, "trained_model.pkl")

                st.success("Model trained successfully!")
                st.write(f"Accuracy: `{metrics['accuracy']:.3f}`")

                # Confusion matrix
                fig, ax = plt.subplots()
                ax.matshow(cm, cmap="Blues")
                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        ax.text(j, i, cm[i][j], va='center', ha='center')
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Training error: {e}")

    # PDF Report
    if os.path.exists("trained_model.pkl"):
        st.write("### Generate Full PDF Report")
        if st.button("Generate"):
            try:
                path = generate_full_report(
                    pdf_path="model_report.pdf",
                    model_path="trained_model.pkl",
                    data_path="loan_data.csv",
                    sensitive_attribute=None,
                )
                with open(path, "rb") as f:
                    st.download_button("⬇ Download Report", f, "model_report.pdf")
            except Exception as e:
                st.error(f"PDF error: {e}")


# ===============================
# FAIRNESS PAGE
# ===============================
elif action == "🧪 Fairness Evaluation":
    try:
        evaluate_fairness()
    except Exception as e:
        st.error(f"Fairness error: {e}")


# ===============================
# EXPLAINABILITY PAGE
# ===============================
elif action == "🔍 Explainability":
    try:
        result = explain_model_ui()
        if result:
            shap_values, feature_names, row_data = result
            explanation = generate_natural_language_explanation(
                shap_values, feature_names, row_data
            )
            st.markdown("---")
            st.write(explanation)

    except Exception as e:
        st.error(f"Explainability error: {e}")

elif action == "📈 Feature Insights":
    st.subheader("📈 Feature Insights")

    try:
        import shap
        model = safe_load_model()
        df = safe_load_data()

        if model is None or df is None:
            st.warning("Please upload a dataset and train the model first.")
        else:
            target = df.columns[-1]
            X = df.drop(columns=[target])

            pre = model.named_steps["preprocessor"]
            clf = model.named_steps["classifier"]

            # Transform dataset
            X_trans = pre.transform(X)

            # Convert sparse to array
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()

            # Get SHAP values
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_trans)

            # If binary classifier → use class 1 SHAP
            shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

            # Ensure correct shape
            shap_vals = np.array(shap_vals).reshape(len(X_trans), -1)

            # ---- FIX: Proper feature name handling ----
            try:
                feature_names = pre.get_feature_names_out()
            except:
                feature_names = np.array([f"Feature_{i}" for i in range(shap_vals.shape[1])])

            # If mismatch → force correction
            if len(feature_names) != shap_vals.shape[1]:
                feature_names = np.array([f"Feature_{i}" for i in range(shap_vals.shape[1])])

            # Compute final importances
            shap_mean = np.mean(np.abs(shap_vals), axis=0)

            # Build dataframe
            feat_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": shap_mean
            }).sort_values("Importance", ascending=False)

            st.write("### 🔝 Top 20 Features")
            st.dataframe(feat_df.head(20))

            fig = px.bar(
                feat_df.head(20),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top SHAP Feature Importances"
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Feature Insights Error: {e}")

# ===============================
# MODEL RISK SCORE
# ===============================
elif action == "🚦 Model Risk Score":
    try:
        df = safe_load_data()
        model = safe_load_model()

        if df is None or model is None:
            st.warning("Train a model first.")
        else:
            target = df.columns[-1]
            X = df.drop(columns=[target])
            y = df[target]

            try:
                classes = model.named_steps["classifier"].classes_
                le = LabelEncoder()
                le.fit(classes)
                y_enc = le.transform(y)
            except:
                y_enc = y

            y_pred = model.predict(X)
            acc = accuracy_score(y_enc, y_pred)

            sensitive = [c for c in df.columns if df[c].dtype == object]
            bias = []
            for col in sensitive:
                g = df.groupby(col)[target].mean()
                if g.max() - g.min() > 0.25:
                    bias.append(col)

            risk = 100 - (acc * 60) - (len(bias) * 10)
            risk = max(0, min(100, risk))

            st.write(f"### Accuracy: `{acc:.3f}`")
            st.write(f"### Fairness Flags: `{bias or 'None'}`")
            st.write(f"### 🚦 Final Risk Score: **{risk:.0f}/100**")

    except Exception as e:
        st.error(f"Risk Score Error: {e}")


# ===============================
# PREDICT PAGE
# ===============================
elif action == "🧮 Predict on New Data":
    df = safe_load_data()
    model = safe_load_model()

    if df is None or model is None:
        st.warning("Train a model first.")
    else:
        target = df.columns[-1]
        features = df.drop(columns=[target]).columns

        st.write("Enter values for prediction:")
        user_input = {}

        for f in features:
            if pd.api.types.is_numeric_dtype(df[f]):
                user_input[f] = st.number_input(f, value=float(df[f].median()))
            else:
                user_input[f] = st.selectbox(f, df[f].unique().tolist())

        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])
            pred = model.predict(input_df)[0]

            st.success(f"Prediction: **{pred}**")


# ===============================
# VIEW DATASET
# ===============================
elif action == "📦 View Dataset":
    df = safe_load_data()
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.dataframe(df)
        st.write(df.describe(include="all"))
