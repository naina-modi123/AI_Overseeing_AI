# report_generator.py
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import joblib
import os


# --------------------------------------------------------
# Helper: Confusion Matrix Plot
# --------------------------------------------------------
def generate_confusion_matrix_plot(model, X, y_numeric):
    """
    Generates and saves confusion matrix plot.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y_numeric, y_pred)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    path = "confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path


# --------------------------------------------------------
# Helper: Fairness comparison plot
# --------------------------------------------------------
def generate_fairness_plot(df, sensitive_attribute, y_numeric):
    """
    Plots average label rate per sensitive attribute group.
    """
    df_copy = df.copy()
    df_copy["target"] = y_numeric

    fairness = df_copy.groupby(sensitive_attribute)["target"].mean()

    plt.figure(figsize=(4, 3))
    fairness.plot(kind="bar", color="#4C9AFF")
    plt.ylabel("Positive Prediction Rate")
    plt.title(f"Fairness Comparison by {sensitive_attribute}")
    plt.tight_layout()

    path = "fairness_plot.png"
    plt.savefig(path)
    plt.close()
    return path


# --------------------------------------------------------
# Main Report Generator
# --------------------------------------------------------
def generate_full_report(pdf_path, model_path, data_path, sensitive_attribute=None):
    """
    Creates a full PDF report including:
    - Dataset preview
    - Model accuracy
    - Confusion matrix
    - Optional fairness analysis
    """

    # Load model & data
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    flow = []
    styles = getSampleStyleSheet()

    # PDF Setup
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)

    # -------------------------------
    # PDF Title
    # -------------------------------
    flow.append(Paragraph("<b>AI Model Evaluation Report</b>", styles["Title"]))
    flow.append(Spacer(1, 20))

    # -------------------------------
    # Dataset summary
    # -------------------------------
    flow.append(Paragraph("<b>Dataset Overview</b>", styles["Heading2"]))
    flow.append(Paragraph(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}", styles["Normal"]))
    flow.append(Spacer(1, 10))

    # -------------------------------
    # Model Accuracy
    # -------------------------------
    target_col = df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Convert y_true to numeric (IMPORTANT FIX)
    if y.dtype == object:
        y_numeric, uniques = pd.factorize(y)
    else:
        y_numeric = y

    y_pred = model.predict(X)
    acc = accuracy_score(y_numeric, y_pred)

    flow.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))
    flow.append(Paragraph(f"Accuracy: {acc:.3f}", styles["Normal"]))
    flow.append(Spacer(1, 12))

    # -------------------------------
    # Confusion Matrix Section
    # -------------------------------
    flow.append(Paragraph("<b>Confusion Matrix</b>", styles["Heading2"]))
    cm_path = generate_confusion_matrix_plot(model, X, y_numeric)
    flow.append(Image(cm_path, width=350, height=260))
    flow.append(Spacer(1, 20))

    # -------------------------------
    # Optional Fairness Analysis
    # -------------------------------
    if sensitive_attribute is not None and sensitive_attribute in df.columns:
        flow.append(Paragraph("<b>Fairness Analysis</b>", styles["Heading2"]))
        fair_path = generate_fairness_plot(df, sensitive_attribute, y_numeric)
        flow.append(Image(fair_path, width=350, height=260))
        flow.append(Spacer(1, 20))

    # Build PDF
    doc.build(flow)

    return pdf_path
