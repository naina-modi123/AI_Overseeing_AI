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

        try:
            group_stats = df.groupby(sensitive)[target].mean()
        except Exception as e:
            st.error(f"Fairness computation failed: {e}")
            return

        st.write("### 📊 Average target rate per group")
        st.dataframe(group_stats)

        fig, ax = plt.subplots()
        group_stats.plot(kind="bar", color="#4C9AFF", ax=ax)
        ax.set_title("Average Target Outcome by Group")
        ax.set_ylabel("Mean Outcome")
        st.pyplot(fig)

        st.success("Fairness evaluation completed!")


# ============================================================
# 2️⃣ SHAP EXPLAINABILITY (RETURNS VALUES FOR NLG)
# ============================================================
def explain_model_ui():
    import shap

    st.subheader("🔍 Model Explainability (SHAP)")

    try:
        model = joblib.load("trained_model.pkl")
    except:
        st.error("⚠️ No trained model found. Train a model first.")
        return None

    try:
        df = pd.read_csv("loan_data.csv")
    except:
        st.error("⚠️ Dataset 'loan_data.csv' not found.")
        return None

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = X.columns

    st.write("### Select a row to explain")
    row_idx = st.number_input(
        "Row index", min_value=0, max_value=len(X) - 1, value=0
    )

    st.info("Computing SHAP values... ⏳")

    try:
        X_transformed = preprocessor.transform(X)

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
    except Exception as e:
        st.error(f"SHAP failed: {e}")
        return None

    pred = model.predict(X.iloc[[row_idx]])[0]
    st.success(f"### 🔮 Prediction for row {row_idx}: **{pred}**")

    st.write("### 📊 SHAP Summary Plot")
    fig_sum = plt.figure(figsize=(8, 4))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    st.pyplot(fig_sum)

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
        st.warning("Force plot is not supported.")

    # ⭐ RETURN VALUES FOR NLG
    return shap_values[pred][row_idx], feature_names, X.iloc[row_idx].values


# ============================================================
# 3️⃣ NATURAL LANGUAGE EXPLANATION
# ============================================================
def generate_natural_language_explanation(shap_values, feature_names, row_data):
    shap_abs = [abs(val) for val in shap_values]
    top_indices = sorted(range(len(shap_abs)), key=lambda i: shap_abs[i], reverse=True)[:3]

    explanation_parts = []

    for idx in top_indices:
        feature = feature_names[idx]
        value = row_data[idx]
        impact = shap_values[idx]

        if impact > 0:
            explanation_parts.append(
                f"- **{feature}** increased the prediction because its value ({value}) had a positive influence."
            )
        else:
            explanation_parts.append(
                f"- **{feature}** decreased the prediction because its value ({value}) had a negative influence."
            )

    final_text = (
        "### 🗣️ Natural Language Explanation\n"
        "Based on SHAP analysis, the model made this prediction because:\n\n"
        + "\n".join(explanation_parts)
        + "\n\nThese features had the strongest impact."
    )

    return final_text


# ============================================================
# 4️⃣ MODEL RISK SCORING (UPDATED & FIXED)
# ============================================================
def calculate_model_risk_score(model, df, shap_values, feature_names):

    risk_components = {}

    # 1️⃣ Accuracy Risk
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Fix: Encode y if strings ('Y','N','Yes','No')
    if y.dtype == object:
        unique_vals = list(y.unique())
        mapping = {unique_vals[i]: i for i in range(len(unique_vals))}
        y = y.map(mapping)

    acc = model.score(X, y)
    risk_components["accuracy"] = 1 - acc

    # 2️⃣ Fairness Risk
    fairness_risk = 0
    for col in df.columns:
        if df[col].dtype != object and df[col].nunique() > 2:
            continue
        try:
            rates = df.groupby(col)[target_col].mean()
            diff = abs(rates.max() - rates.min())
            fairness_risk = max(fairness_risk, diff)
        except:
            pass
    risk_components["fairness"] = fairness_risk

    # 3️⃣ Explainability Risk
    shap_mean = np.mean([abs(s) for s in shap_values])
    explain_risk = 1 / (1 + shap_mean)
    risk_components["explainability"] = explain_risk

    # 4️⃣ Class Imbalance Risk
    imbalance = df[target_col].value_counts(normalize=True).max()
    risk_components["imbalance"] = imbalance - 0.5

    # ⭐ FINAL SCORE
    final_score = (
        0.35 * risk_components["fairness"] +
        0.30 * risk_components["accuracy"] +
        0.20 * risk_components["explainability"] +
        0.15 * risk_components["imbalance"]
    )

    final_score = min(max(final_score, 0), 1)

    if final_score <= 0.33:
        level = "🟢 LOW RISK"
    elif final_score <= 0.66:
        level = "🟡 MEDIUM RISK"
    else:
        level = "🔴 HIGH RISK"

    return final_score, level, risk_components
