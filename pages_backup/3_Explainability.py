import streamlit as st
from evaluator import explain_model

st.title("🔍 Explainability Dashboard")

try:
    explain_model()
except Exception as e:
    st.error(f"Error: {e}")
