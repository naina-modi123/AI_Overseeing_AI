# evaluator.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1️⃣ FAIRNESS EVALUATION
# ============================================================
def evaluate_fairness():
    st.subheader("🧪 Fairness Evaluation Tool")

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------
    try:
        df = pd.read_csv("loan_data.csv")
    except:
        st.error("⚠️ Dataset not found. Upload a CSV in 'Train Model' page first.")
        return

    st.write("### Select sensitive attribute")
    sensitive = st.selectbox("Sensitive Attribute (e.g., Gender)", options=df.columns)

    st.write("### Select target column")
    target = st.selectbox("Target Column", options=df.columns)

    if st.button("Run Fairness Analysis"):
        st.info("Computing fairness metrics...")

        if sensitive not in df.columns or target not in df.columns:
            st.error(f"Columns are missing: {sensitive}, {target}")
            return

        # -----------------------------------------
        # Compute fairness metrics
        # -----------------------------------------
        try:
            group_stats = df.groupby(sensitive)[target].mean()
        except Exception as e:
            st.error(f"Fairness computation failed: {e}")
            return

        st.write("### 📊 Average target rate per group")
        st.dataframe(group_stats)

        # Bar plot
        fig, ax = plt.subplots()
        group_stats.plot(kind="bar", color="#4C9AFF", ax=ax)
        ax.set_title("Average Target Outcome by Group")
        ax.set_ylabel("Mean Outcome")
        st.pyplot(fig)

        st.success("Fairness evaluation completed!")


# ============================================================
# 2️⃣ SHAP EXPLAINABILITY
# ============================================================
def explain_model_ui():
    import shap  # imported here to prevent errors if unavailable

    st.subheader("🔍 Model Explainability (SHAP)")

    # -----------------------------------------
    # Load model
    # -----------------------------------------
    try:
        model = joblib.load("trained_model.pkl")
    except:
        st.error("⚠️ No trained model found. Train a model first.")
        return

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------
    try:
        df = pd.read_csv("loan_data.csv")
    except:
        st.error("⚠️ Dataset 'loan_data.csv' not found.")
        return

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])

    # -----------------------------------------
    # Get feature names after preprocessing
    # -----------------------------------------
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = X.columns  # fallback

    # -----------------------------------------
    # Select row to explain
    # -----------------------------------------
    st.write("### Select a row to explain")
    row_idx = st.number_input(
        "Row index",
        min_value=0,
        max_value=len(X) - 1,
        value=0
    )

    st.info("Computing SHAP values... ⏳")

    # -----------------------------------------
    # SHAP computation
    # -----------------------------------------
    try:
        X_transformed = preprocessor.transform(X)

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
    except Exception as e:
        st.error(f"SHAP failed: {e}")
        return

    # -----------------------------------------
    # Prediction
    # -----------------------------------------
    pred = model.predict(X.iloc[[row_idx]])[0]
    st.success(f"### 🔮 Model Prediction for row {row_idx}: **{pred}**")

    # -----------------------------------------
    # Summary Plot
    # -----------------------------------------
    st.write("### 📊 SHAP Summary Plot")
    fig_sum = plt.figure(figsize=(8, 4))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    st.pyplot(fig_sum)

    # -----------------------------------------
    # Waterfall plot (best explanation)
    # -----------------------------------------
    st.write("### 🌊 SHAP Waterfall Plot")

    try:
        fig_wf = plt.figure(figsize=(8, 4))
        waterfall_obj = shap.Explanation(
            values=shap_values[pred][row_idx],
            base_values=explainer.expected_value[pred],
            data=X_transformed[row_idx],
            feature_names=feature_names
        )
        shap.plots.waterfall(waterfall_obj, show=False)
        st.pyplot(fig_wf)
    except Exception as e:
        st.warning(f"Waterfall plot not supported: {e}")

    # -----------------------------------------
    # Force plot (HTML)
    # -----------------------------------------
    st.write("### ⚡ SHAP Force Plot")

    try:
        force = shap.force_plot(
            explainer.expected_value[pred],
            shap_values[pred][row_idx],
            X_transformed[row_idx],
            matplotlib=False
        )
        st.components.v1.html(force.html(), height=300)
    except:
        st.warning("Force plot is not supported in this environment.")
