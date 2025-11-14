
# AI_Overseeing_AI — Starter Project for "Balancing AI: Innovation and Ethics"
This repository contains a beginner-friendly prototype of an **AI Oversight** tool, inspired by your PPT idea.
It trains a simple logistic regression model on a loan dataset, evaluates fairness, and attempts to show SHAP explanations.

## Files
- `model.py` — trains and saves a simple logistic regression model from `loan_data.csv`.
- `evaluator.py` — provides fairness evaluation and explainability helpers.
- `app.py` — Streamlit app to interactively train and evaluate.
- `loan_data.csv` — a small synthetic sample dataset (included).
- `requirements.txt` — Python packages to install.
- `README.md` — this file.

## Quick start (local)
1. Create a virtual environment (recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## Notes
- SHAP and fairlearn may need extra time to install; if they fail, the app will still work but with limited features.
- Use your real dataset by uploading a CSV with a `Loan_Status` column containing `Y`/`N` values.
