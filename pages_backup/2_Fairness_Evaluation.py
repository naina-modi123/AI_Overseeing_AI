import streamlit as st
from evaluator import evaluate_fairness

st.title("🧪 Fairness Evaluation")

try:
    evaluate_fairness()
except Exception as e:
    st.error(f"Error: {e}")
