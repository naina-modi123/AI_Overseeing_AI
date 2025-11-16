# evaluator.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap


# =============================================================
# 1️⃣ FAIRNESS EVALUATION  (works for numeric / encoded binary target)
# =============================================================
def evaluate_fairness():
    st.subheader("🧪 Fairness Evaluation Tool")

    try:
        df = pd.read_csv("loan_data.csv")
    except Exception:
        st.error("⚠️ Dataset not found. Upload in Train Model page.")
        return

    st.write("### Select sensitive attribute")
    sensitive = st.selectbox("Sensitive Attribute", df.columns)

    st.write("### Select target column")
    target = st.selectbox("Target Column", df.columns)

    if st.button("Run Fairness Analysis"):
        st.info("Computing fairness metrics...")

        try:
            # Convert target to numeric if needed (safe)
            y = df[target]
            if y.dtype == object:
                df[target] = y.astype("category").cat.codes

            group_stats = df.groupby(sensitive)[target].mean()

        except Exception as e:
            st.error(f"Fairness computation failed: {e}")
            return

        st.write("### 📊 Average target outcome per group")
        st.dataframe(group_stats.reset_index())

        fig, ax = plt.subplots()
        group_stats.plot(kind="bar", color="#4C9AFF", ax=ax)
        ax.set_title("Average Target Outcome by Group")
        ax.set_ylabel("Mean Outcome")
        st.pyplot(fig)

        st.success("Fairness evaluation completed!")


# =============================================================
# 2️⃣ SHAP EXPLAINABILITY (returns: row_shap_vector, feature_names, raw_row)
# Robust handling for different shap output shapes / versions.
# =============================================================
def explain_model_ui():
    st.subheader("🔍 Model Explainability (SHAP)")

    # load model
    try:
        model = joblib.load("trained_model.pkl")
    except Exception:
        st.error("⚠️ No trained model found. Train a model first.")
        return None

    # load data
    try:
        df = pd.read_csv("loan_data.csv")
    except Exception:
        st.error("⚠️ Dataset 'loan_data.csv' not found.")
        return None

    target = df.columns[-1]
    X = df.drop(columns=[target])

    try:
        pre = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]
    except Exception:
        st.error("Model pipeline must contain 'preprocessor' and 'classifier' steps.")
        return None

    # get feature names (fallback to X.columns)
    try:
        feature_names = list(pre.get_feature_names_out())
    except Exception:
        feature_names = list(X.columns)

    row_idx = st.number_input("Row index", min_value=0, max_value=max(0, len(X) - 1), value=0)
    st.info("Computing SHAP values...")

    # transform features for SHAP
    try:
        Xt = pre.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
    except Exception as e:
        st.error(f"Preprocessor transform failed: {e}")
        return None

    # compute shap values with TreeExplainer (works for tree models)
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values_raw = explainer.shap_values(Xt)
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        return None

    # Normalize shap_values to a 2D array: (n_samples, n_features)
    class_idx = 0  # which class we choose (for multiclass/binary choose positive/class 1 if available)
    sv = None
    try:
        # list case: common for multiclass/tree SHAP
        if isinstance(shap_values_raw, list):
            # If binary/multiclass this is list[class][n_samples, n_features]
            if len(shap_values_raw) > 1:
                class_idx = 1 if len(shap_values_raw) > 1 else 0
                sv = np.array(shap_values_raw[class_idx])
            else:
                sv = np.array(shap_values_raw[0])

        # numpy array
        elif isinstance(shap_values_raw, np.ndarray):
            # possible shapes:
            # - (n_samples, n_features) -> fine
            # - (n_outputs, n_samples, n_features) -> choose class index 1 if exists
            # - (n_samples, n_features, n_outputs) -> choose last axis index 1 if exists
            if shap_values_raw.ndim == 2:
                sv = shap_values_raw
            elif shap_values_raw.ndim == 3:
                a, b, c = shap_values_raw.shape
                # if first dim equals n_samples -> (n_samples, n_features, n_outputs)
                if a == Xt.shape[0]:
                    # choose output index 1 if possible else 0
                    out_idx = 1 if c > 1 else 0
                    sv = shap_values_raw[:, :, out_idx]
                    class_idx = out_idx
                # else if second dim equals n_samples -> (n_outputs, n_samples, n_features)
                elif b == Xt.shape[0]:
                    cls = 1 if a > 1 else 0
                    sv = shap_values_raw[cls]
                    class_idx = cls
                else:
                    # fallback: try to reshape to (n_samples, n_features)
                    try:
                        sv = shap_values_raw.reshape(Xt.shape[0], -1)
                    except Exception:
                        sv = np.array(shap_values_raw)
            else:
                # unexpected dims -> try to coerce
                sv = shap_values_raw.reshape(Xt.shape[0], -1)
        else:
            sv = np.array(shap_values_raw)

    except Exception as e:
        st.error(f"Failed to normalize SHAP output: {e}")
        return None

    # ensure sv is 2D and matches X shape
    try:
        sv = np.asarray(sv)
        if sv.ndim != 2 or sv.shape[0] != Xt.shape[0]:
            st.warning(f"SHAP output has unexpected shape {sv.shape}; attempting transpose/reshape.")
            # try transpose if that fixes it
            if sv.ndim == 2 and sv.shape[1] == Xt.shape[0]:
                sv = sv.T
    except Exception as e:
        st.error(f"SHAP postprocessing failed: {e}")
        return None

    # prediction for the selected row (show to user)
    try:
        pred = model.predict(X.iloc[[row_idx]])[0]
        st.success(f"### 🔮 Prediction for row {row_idx}: **{pred}**")
    except Exception as e:
        st.warning(f"Could not compute model prediction for row: {e}")

    # SHAP summary plot
    try:
        fig = plt.figure(figsize=(8, 4))
        shap.summary_plot(sv, Xt, feature_names=feature_names, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP summary plot not available: {e}")

    # Waterfall plot: ensure we pass a 1D explanation vector
    try:
        sv_row = sv[row_idx]

        if sv_row.ndim != 1:
            sv_row = np.ravel(sv_row)

        # pick base value matching class_idx
        base_val = explainer.expected_value
        try:
            # expected_value might be list/array for multiclass
            if isinstance(base_val, (list, tuple, np.ndarray)):
                if len(base_val) > class_idx:
                    base_val = base_val[class_idx]
                else:
                    base_val = base_val[0]
        except Exception:
            pass

        explanation = shap.Explanation(
            values=sv_row,
            base_values=base_val,
            data=Xt[row_idx],
            feature_names=feature_names,
        )

        fig_w = plt.figure(figsize=(8, 4))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig_w)

    except Exception as e:
        st.warning(f"Waterfall plot not supported: {e}")

    # Return a 1D row shap vector + feature_names + original row for NLG
    try:
        row_shap = np.ravel(sv[row_idx])
        return row_shap, feature_names, X.iloc[row_idx].values
    except Exception:
        return None


# ============================================================
# 3️⃣ NATURAL LANGUAGE EXPLANATION (NLG)
# ============================================================
def generate_natural_language_explanation(shap_values, feature_names, row_data):
    """
    shap_values: 1D array for the selected row
    feature_names: list-like of feature names (length == len(shap_values))
    row_data: raw row values (pandas Series or numpy array)
    """

    if shap_values is None:
        return "No SHAP values available to generate explanation."

    shap_vals = np.asarray(shap_values)
    # safety: ensure lengths match
    N = len(shap_vals)
    fnames = list(feature_names)[:N]
    row_vals = list(row_data)[:N]

    # top-k by absolute importance
    top_k = 3
    idxs = sorted(range(N), key=lambda i: abs(shap_vals[i]), reverse=True)[:top_k]

    parts = []
    for i in idxs:
        fname = fnames[i] if i < len(fnames) else f"feature_{i}"
        fval = row_vals[i] if i < len(row_vals) else "N/A"
        impact = float(shap_vals[i])
        direction = "increased" if impact > 0 else "decreased"
        parts.append(f"- **{fname}** ({fval}) {direction} the prediction (contribution ≈ {impact:.3f}).")

    final_text = (
        "### 🗣️ Natural Language Explanation\n\n"
        "The model's prediction was influenced most strongly by the following features:\n\n"
        + "\n".join(parts)
        + "\n\nThese are the top contributors according to SHAP."
    )

    return final_text
