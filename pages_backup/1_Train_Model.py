import streamlit as st
import pandas as pd
from model import train_model

st.title("📊 Train Model")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    with open("loan_data.csv", "wb") as f:
        f.write(uploaded.getbuffer())

    st.success("CSV uploaded successfully!")

    try:
        model, X_test, y_test, acc = train_model("loan_data.csv")
        st.success(f"Model trained successfully! Accuracy: **{acc:.3f}**")
    except Exception as e:
        st.error(f"Error: {e}")
