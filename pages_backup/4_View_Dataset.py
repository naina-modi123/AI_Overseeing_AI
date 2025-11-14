import streamlit as st
import pandas as pd

st.title("📦 Dataset Viewer")

try:
    df = pd.read_csv("loan_data.csv")
    st.dataframe(df)
except:
    st.warning("Please upload and train a model first.")
