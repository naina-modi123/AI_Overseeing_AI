# ============================================
# app.py — Two Themes + Auto-Bias Correction + AI Assistant
# CLEAN FINAL VERSION (no login) — FIXED
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# ML helpers
from model import train_model
from evaluator import (
    evaluate_fairness,
    explain_model_ui,
    generate_natural_language_explanation,
)

from report_generator import generate_full_report
import shap

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="AI Overseeing AI", layout="wide")

# -----------------------
# Theme system (NEON + PASTEL)
# -----------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def inject_theme_css():
    if st.session_state.theme == "dark":
        css = """
        <style>
        .stApp { background: radial-gradient(circle at top, #1a0033, #000011 60%, #000000); color: #E9E9FF !important; }
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, #120022, #000010); color: #e8d8ff; }
        .main-title{ text-align:center; font-size:46px; font-weight:800; margin-top:-5px; text-shadow:0 0 18px #a86aff; color:#e7d9ff !important; }
        .sub-title{ text-align:center; color:#c5b7ff; margin-top:-15px; margin-bottom:25px; }

        .glass-card { 
            background: rgba(20,20,35,0.55); 
            border:1px solid rgba(120,60,255,0.3); 
            backdrop-filter: blur(12px); 
            box-shadow:0 0 22px rgba(120,60,255,0.35);
            border-radius:18px; padding:22px; 
            margin-bottom:20px; transition:0.25s ease; 
        }
        .glass-card:hover{ transform: translateY(-5px); box-shadow:0 0 35px rgba(160,90,255,0.9); }

        .card-icon { font-size:36px; margin-bottom:10px; }
        .card-title { font-size:20px; font-weight:700; margin-bottom:6px; color:white; }
        .card-desc { font-size:14px; color:#cfc7ff; }

        .stButton>button { 
            background: linear-gradient(90deg,#6c25ff,#b400ff); 
            color:white; border:none; border-radius:10px; 
            box-shadow:0 0 12px rgba(160,60,255,0.7);
        }
        .stButton>button:hover { box-shadow:0 0 20px rgba(200,80,255,1); }

        </style>
        """
    else:
        css = """
        <style>
        .stApp { background: linear-gradient(135deg,#ffe5f1,#ffdfe8,#fff7fb); color:#43223d !important; }
        section[data-testid="stSidebar"] { background:#ffeef6; color:#43223d; }
        .main-title{ text-align:center; font-size:46px; font-weight:800; margin-top:-5px; color:#b4236d !important; text-shadow:0 0 12px rgba(255,140,180,0.5); }
        .sub-title{ text-align:center; color:#904061; margin-top:-15px; margin-bottom:25px; }

        .glass-card{ 
            background: rgba(255,255,255,0.60); 
            border:1px solid rgba(255,170,200,0.45);
            backdrop-filter: blur(14px);
            border-radius:18px; padding:22px;
            box-shadow:0 6px 20px rgba(255,150,190,0.25);
            margin-bottom:20px; transition:0.3s ease;
        }
        .glass-card:hover{ transform: translateY(-5px); box-shadow:0 8px 24px rgba(255,140,180,0.5); }

        .card-icon { font-size:36px; margin-bottom:10px; }
        .card-title { font-size:20px; font-weight:700; margin-bottom:6px; color:#6b1d40; }
        .card-desc { font-size:14px; color:#8a4d6b; }

        .stButton>button { 
            background: linear-gradient(90deg,#ff8abb,#ffb1d6);
            color:white; border:none; border-radius:10px;
            box-shadow:0 4px 14px rgba(255,120,170,0.4);
        }
        .stButton>button:hover { box-shadow:0 6px 22px rgba(255,110,180,0.6); }

        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# Theme selector (fixed single-click now)
with st.sidebar:
    theme_choice = st.selectbox(
        "Select Theme", ["dark", "light"], 
        index=0 if st.session_state.theme == "dark" else 1
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
inject_theme_css()


# -----------------------
# Sidebar navigation
# -----------------------
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
    ],
)

# -----------------------
# Loaders
# -----------------------
def safe_load_data():
    if os.path.exists("loan_data.csv"):
        try: return pd.read_csv("loan_data.csv")
        except: return None
    return None

def safe_load_model():
    if os.path.exists("trained_model.pkl"):
        try: return joblib.load("trained_model.pkl")
        except: return None
    return None


# -----------------------
# Auto-bias correction (fixed: avoid resample n_samples=0)
# -----------------------
def auto_bias_resample(df, sensitive_col, target_col):
    df = df.copy()
    frames = []
    groups = df.groupby(sensitive_col)
    overall_rate = df[target_col].mean()

    for grp, gdf in groups:
        # expected numeric target (0/1)
        pos = gdf[gdf[target_col] == 1]
        neg = gdf[gdf[target_col] == 0]
        grp_len = len(gdf)
        grp_rate = 0 if grp_len == 0 else pos.shape[0] / max(1, grp_len)

        # undersampled (raise positives)
        if grp_rate < overall_rate and len(pos) > 0:
            desired = int(round(overall_rate * grp_len))
            need = desired - len(pos)
            if need > 0:
                pos_up = resample(pos, replace=True, n_samples=need, random_state=42)
                frames.append(pd.concat([gdf, pos_up], ignore_index=True))
            else:
                frames.append(gdf)
        # oversampled (reduce positives)
        elif grp_rate > overall_rate and len(pos) > 0:
            desired = int(round(overall_rate * grp_len))
            keep = max(0, desired)
            if keep >= 1:
                pos_down = resample(pos, replace=False, n_samples=min(len(pos), keep), random_state=42)
                frames.append(pd.concat([pos_down, neg], ignore_index=True))
            else:
                # cannot sample zero positives: just keep group as-is
                frames.append(gdf)
        else:
            frames.append(gdf)

    if len(frames) == 0:
        return df.copy()
    new_df = pd.concat(frames, ignore_index=True)
    # guard: if new_df has zero rows, return original
    if new_df.shape[0] == 0:
        return df.copy()
    return new_df.sample(frac=1, random_state=42).reset_index(drop=True)


# -----------------------
# HOME
# -----------------------
if action == "🏠 Home":
    st.markdown("<h1 class='main-title'>Responsible AI Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Train models • Check fairness • Explain decisions</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon">📊</div>
            <div class="card-title">Train a Model</div>
            <div class="card-desc">Upload a dataset & train instantly.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon">🧪</div>
            <div class="card-title">Fairness Evaluation</div>
            <div class="card-desc">Evaluate demographic bias.</div>
        </div>
        """, unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon">🔍</div>
            <div class="card-title">Explainability</div>
            <div class="card-desc">Understand predictions with SHAP.</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon">🧮</div>
            <div class="card-title">Prediction Tool</div>
            <div class="card-desc">Predict outcomes from inputs.</div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------
# TRAIN + AUTO-BIAS
# -----------------------
elif action == "📊 Train Model (CSV)":
    st.header("📊 Train a Machine Learning Model")

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded:
        with open("loan_data.csv", "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("Dataset uploaded successfully!")

    df = safe_load_data()
    if df is not None:
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        st.markdown("---")
        st.write("### 🎯 Auto-Bias Correction")

        # Auto-detect possible sensitive attributes
        sensitive_cols = [c for c in df.columns if df[c].dtype == object or df[c].nunique() <= 10]

        if sensitive_cols:
            sensitive = st.selectbox("Sensitive Attribute", ["(none)"] + sensitive_cols)
            target = st.selectbox("Target Column", df.columns.tolist(), index=len(df.columns)-1)

            # -------------------------------
            # Helper: robust numeric transform
            # -------------------------------
            def _make_numeric(series):
                if series.dtype == object:
                    try:
                        le = LabelEncoder()
                        return le.fit_transform(series)
                    except Exception:
                        return series.astype("category").cat.codes
                return series

            if sensitive != "(none)" and st.button("Preview Bias Correction"):
                try:
                    # work on copy, convert target to numeric if needed
                    temp_df = df.copy()
                    temp_df[target] = _make_numeric(temp_df[target])

                    # ensure target is numeric with exactly two classes (0/1) for our simple resampling logic
                    uniq = temp_df[target].dropna().unique()
                    if len(uniq) < 2:
                        st.error("Target must have at least 2 classes for bias correction.")
                    else:
                        # Before correction
                        before = temp_df.groupby(sensitive)[target].mean().rename("before_rate")
                        st.write("Before Bias Rates")
                        st.dataframe(before.reset_index())

                        # Perform correction
                        new_df = auto_bias_resample(temp_df, sensitive, target)

                        # After correction
                        after = new_df.groupby(sensitive)[target].mean().rename("after_rate")
                        st.write("After Bias Rates")
                        st.dataframe(after.reset_index())

                except Exception as e:
                    st.error(f"Bias Correction Error: {e}")

                if st.button("Retrain on Corrected Data"):
                    try:
                        new_df.to_csv("loan_data_bias_corrected.csv", index=False)
                        with st.spinner("Retraining..."):
                            model, *_ = train_model("loan_data_bias_corrected.csv")
                            joblib.dump(model, "trained_model.pkl")
                            st.success("Retrained model saved!")
                    except Exception as e:
                        st.error(f"Retrain failed: {e}")

        st.markdown("---")

        # ---------------------------
        # STANDARD MODEL TRAINING
        # ---------------------------
        if st.button("Train Model"):
            with st.spinner("Training..."):
                try:
                    model, X_test, y_test, acc, metrics, cm = train_model("loan_data.csv")
                    joblib.dump(model, "trained_model.pkl")

                    st.success("Model trained!")
                    st.write("Accuracy:", metrics.get("accuracy", acc))

                    fig, ax = plt.subplots()
                    ax.matshow(cm, cmap="Blues")
                    for i in range(len(cm)):
                        for j in range(len(cm[i])):
                            ax.text(j, i, cm[i][j], ha="center", va="center")
                    st.pyplot(fig)

                except Exception as e:
                    st.error(e)

        # PDF Report
        if os.path.exists("trained_model.pkl"):
            if st.button("Generate PDF Report"):
                try:
                    path = generate_full_report("model_report.pdf", "trained_model.pkl", "loan_data.csv", None)
                    with open(path, "rb") as f:
                        st.download_button("Download Report", f, "model_report.pdf")
                except Exception as e:
                    st.error(e)


# -----------------------
# FAIRNESS PAGE
# -----------------------
elif action == "🧪 Fairness Evaluation":
    try:
        evaluate_fairness()
    except Exception as e:
        st.error(e)

# -----------------------
# EXPLAINABILITY
# -----------------------
elif action == "🔍 Explainability":
    try:
        result = explain_model_ui()
        if isinstance(result, tuple):
            shap_values, feature_names, row = result
            nlg = generate_natural_language_explanation(shap_values, feature_names, row)
            st.markdown("---")
            st.write(nlg)
    except Exception as e:
        st.error(e)

# -----------------------
# FEATURE INSIGHTS
# -----------------------
elif action == "📈 Feature Insights":
    st.header("📈 Feature Insights")

    try:
        model = safe_load_model()
        df = safe_load_data()
        if model is None or df is None:
            st.warning("Train a model first.")
        else:
            target = df.columns[-1]
            X = df.drop(columns=[target])

            # Preprocessor output
            pre = None
            clf = None
            try:
                pre = model.named_steps.get("preprocessor")
                clf = model.named_steps.get("classifier")
            except Exception:
                # pipeline may not have named_steps; try to use model directly
                pass

            if pre is None or clf is None:
                st.error("Model pipeline must contain 'preprocessor' and 'classifier' named steps.")
            else:
                Xt = pre.transform(X)
                if hasattr(Xt, "toarray"):
                    Xt = Xt.toarray()

                # Get feature names safely
                try:
                    feature_names = pre.get_feature_names_out()
                except:
                    feature_names = [f"f{i}" for i in range(Xt.shape[1])]

                # Compute SHAP robustly (try kernel, fallback to tree)
                try:
                    # Try KernelExplainer (works for blackbox) — but is heavy
                    f = lambda inp: clf.predict_proba(inp)[:, 1]
                    # pick small background to speed up
                    background = Xt[: min(50, len(Xt))]
                    explainer = shap.KernelExplainer(f, background)
                    # limit samples to keep UI responsive
                    sv = explainer.shap_values(Xt[: min(200, len(Xt))], nsamples=100)
                except Exception:
                    # Fallback to TreeExplainer for tree models
                    try:
                        explainer = shap.TreeExplainer(clf)
                        sv = explainer.shap_values(Xt)
                        if isinstance(sv, list):
                            sv = sv[1]
                    except Exception as e:
                        st.error(f"SHAP explainability failed: {e}")
                        sv = None

                if sv is None:
                    st.warning("Could not compute feature importances.")
                else:
                    sv = np.array(sv)
                    # mean absolute importance
                    sv_mean = np.abs(sv).mean(axis=0)

                    feat_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": sv_mean
                    }).sort_values("Importance", ascending=False)

                    st.dataframe(feat_df.head(20))

                    fig = px.bar(
                        feat_df.head(20),
                        x="Importance", y="Feature",
                        orientation="h", height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(e)

# -----------------------
# MODEL RISK SCORE
# -----------------------
elif action == "🚦 Model Risk Score":
    st.header("🚦 Model Risk Score")

    try:
        df = safe_load_data()
        model = safe_load_model()

        if df is None or model is None:
            st.warning("Train a model first.")
        else:
            target = df.columns[-1]
            X = df.drop(columns=[target])
            y_true = df[target].copy()

            # Try to get predictions; handle pipeline/classifier differences
            try:
                # If pipeline with named steps
                clf = model.named_steps.get("classifier", None)
                pre = model.named_steps.get("preprocessor", None)
            except Exception:
                clf = None
                pre = None

            if pre is not None and clf is not None:
                Xt = pre.transform(X)
                if hasattr(Xt, "toarray"):
                    Xt = Xt.toarray()
                y_pred = clf.predict(Xt)
            else:
                # fallback
                y_pred = model.predict(X)

            # Align label types
            if pd.api.types.is_numeric_dtype(y_pred) and y_true.dtype == object:
                le = LabelEncoder()
                le.fit(y_true)
                y_true_enc = le.transform(y_true)
                y_pred_enc = y_pred
            elif y_pred.dtype == object and not pd.api.types.is_numeric_dtype(y_true):
                le = LabelEncoder()
                le.fit(y_pred)
                y_pred_enc = le.transform(y_pred)
                y_true_enc = y_true
            else:
                y_pred_enc = y_pred
                y_true_enc = y_true

            # numeric conversion
            try:
                acc = accuracy_score(pd.to_numeric(y_true_enc), pd.to_numeric(y_pred_enc))
            except Exception:
                acc = accuracy_score(y_true_enc, y_pred_enc)

            sensitive = [c for c in df.columns if df[c].dtype == object]
            bias_cols = []
            for c in sensitive:
                try:
                    g = df.groupby(c)[target].apply(lambda x: pd.to_numeric(x, errors="coerce")).mean()
                    if g.max() - g.min() > 0.25:
                        bias_cols.append(c)
                except:
                    pass

            risk = 100 - (acc * 60) - (len(bias_cols) * 10)
            risk = max(0, min(100, risk))

            st.write("Accuracy:", acc)
            st.write("Fairness Flags:", bias_cols or "None")
            st.success(f"Final Model Risk Score: {risk}")

    except Exception as e:
        st.error(e)

# -----------------------
# PREDICT
# -----------------------
elif action == "🧮 Predict on New Data":
    st.header("🧮 Predict")
    df = safe_load_data()
    model = safe_load_model()

    if df is None or model is None:
        st.warning("Train a model first.")
    else:
        target = df.columns[-1]
        features = df.drop(columns=[target]).columns

        user_data = {}
        for f in features:
            if pd.api.types.is_numeric_dtype(df[f]):
                user_data[f] = st.number_input(f, value=float(df[f].median()))
            else:
                user_data[f] = st.selectbox(f, df[f].dropna().unique().tolist())

        if st.button("Predict"):
            try:
                ip = pd.DataFrame([user_data])

                # If pipeline exists with preprocessor + classifier, use them
                try:
                    pre = model.named_steps.get("preprocessor", None)
                    clf = model.named_steps.get("classifier", None)
                except Exception:
                    pre = None
                    clf = None

                if pre is not None and clf is not None:
                    X_ip = pre.transform(ip)
                    if hasattr(X_ip, "toarray"):
                        X_ip = X_ip.toarray()
                    pred = clf.predict(X_ip)[0]
                else:
                    # fallback if model is direct estimator or pipeline not standard
                    pred = model.predict(ip)[0]

                st.success(f"Prediction: {pred}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# -----------------------
# VIEW DATASET
# -----------------------
elif action == "📦 View Dataset":
    df = safe_load_data()
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.dataframe(df)
        st.write(df.describe(include="all"))

# -----------------------
# FLOATING AI ASSISTANT
# -----------------------
if "assistant_open" not in st.session_state:
    st.session_state["assistant_open"] = False

def toggle_assistant():
    st.session_state["assistant_open"] = not st.session_state["assistant_open"]

st.markdown("""
<style>
.fab-button {
    position: fixed;
    right: 20px;
    bottom: 20px;
    z-index: 9999;
}
.assistant-panel {
    position: fixed;
    right: 20px;
    bottom: 80px;
    width: 340px;
    max-height: 55vh;
    overflow-y: auto;
    z-index: 9998;
}
</style>
""", unsafe_allow_html=True)

if st.button("🤖 Assistant", key="float_btn"):
    toggle_assistant()

def assistant_answer(query):
    model = safe_load_model()
    df = safe_load_data()
    q = query.lower().strip()

    if "bias" in q or "fair" in q:
        if df is None:
            return "No dataset loaded."
        categorical = [c for c in df.columns if df[c].dtype == object]
        if not categorical:
            return "No categorical columns to evaluate bias."
        text = "Fairness summary:\n"
        target = df.columns[-1]
        for c in categorical:
            rates = df.groupby(c)[target].mean()
            text += f"- {c}: {rates.to_dict()}\n"
        return text

    if "top feature" in q or "feature import" in q:
        if df is None or model is None:
            return "Train a model first."
        target = df.columns[-1]
        X = df.drop(columns=[target])
        try:
            pre = model.named_steps["preprocessor"]
            clf = model.named_steps["classifier"]
        except Exception:
            return "Model pipeline not found."

        Xt = pre.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.abs(sv).mean(axis=0)

        try: fn = pre.get_feature_names_out()
        except: fn = [f"f{i}" for i in range(len(sv))]

        top = fn[np.argmax(sv)]
        return f"Most influential feature: **{top}**"

    if "explain row" in q:
        if df is None or model is None:
            return "Train a model first."
        try:
            idx = int(q.split()[-1])
            target = df.columns[-1]
            X = df.drop(columns=[target])
            pre = model.named_steps["preprocessor"]
            clf = model.named_steps["classifier"]

            Xt = pre.transform(X)
            if hasattr(Xt, "toarray"): Xt = Xt.toarray()

            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(Xt)
            sv = sv[1] if isinstance(sv, list) else sv
            row_sv = sv[idx]

            try: fn = pre.get_feature_names_out()
            except: fn = [f"Feature_{i}" for i in range(len(row_sv))]

            return generate_natural_language_explanation(row_sv, fn, X.iloc[idx].values)
        except:
            return "Use: 'explain row 0' (example)."

    if "hello" in q or "hi" in q:
        return "Hello! Ask me things like: 'show bias', 'top feature', 'explain row 0'."

    return "I did not understand. Try: 'show bias', 'top feature', 'explain row 0'."

if st.session_state["assistant_open"]:
    st.markdown("<div class='assistant-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Assistant")
    q = st.text_input("Ask:", key="ask_assistant")
    if st.button("Send", key="send_assistant"):
        st.session_state["assistant_response"] = assistant_answer(q)
    if "assistant_response" in st.session_state:
        st.info(st.session_state["assistant_response"])
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
