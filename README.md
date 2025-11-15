# 🌐 AI Overseeing AI Dashboard  
A Responsible AI system for **Model Training, Fairness Evaluation, and Explainability (SHAP)** — all inside an interactive Streamlit interface.

---

## 🚀 Live Demo  
🔗 **Try the App:**  
https://aioverseeingai-vfqdwgumul23ufsf88hfwy.streamlit.app/

---

## 📌 Overview  

**AI Overseeing AI Dashboard** is a full-stack Responsible AI project that allows users to:

- Train ML models with uploaded CSV data  
- Generate accuracy, precision, recall, F1-score  
- Display confusion matrix  
- Perform **fairness analysis** across sensitive attributes  
- Generate **full PDF reports**  
- Explain predictions using **SHAP values**  
- Explore dataset insights and correlations  
- Deploy seamlessly to Streamlit Cloud  

This project ensures **transparency, fairness, and explainability**, which are essential components of modern Responsible AI systems.

---

## 🎯 Features  

### 🔹 1. **Train a Machine Learning Model**  
- Upload any CSV  
- Automatic preprocessing  
- Trains a classification model  
- Displays evaluation metrics  
- Saves the trained model as `.pkl`

### 🔹 2. **Fairness Evaluation**  
- Choose a sensitive attribute (Gender, Married, Education, etc.)  
- Compare model performance across subgroups  
- Identify potential bias  
- Visual fairness distribution plots  
- Error handling for missing or invalid columns  

### 🔹 3. **Explainability (SHAP)**  
- Compute SHAP values  
- Feature importance visualization  
- Row-wise prediction explanations  
- Summary plots  

### 🔹 4. **PDF Report Generator**  
Includes:  
- Model accuracy metrics  
- Confusion matrix  
- Fairness scores  
- SHAP summary  
- Dataset overview  

### 🔹 5. **Dataset Explorer**  
- Preview full dataset  
- Summary statistics  
- Missing value detection  
- Numeric column visualizations  
- Correlation heatmaps  

---

## 🧱 Tech Stack  

| Component | Technology |
|----------|------------|
| Frontend | Streamlit |
| Model Training | Scikit-learn |
| Explainability | SHAP |
| Visualizations | Matplotlib, Seaborn |
| Reporting | ReportLab |
| Deployment | Streamlit Cloud |
| Codebase | Python 3.10+ |

---

## 🧩 Technical Architecture Diagram  

Below is the complete architecture of the **AI Overseeing AI — Responsible AI Evaluation System**.

### **Architecture Flowchart:**


```md
![Technical Architecture](A_flowchart_diagram_depicts_the_technical_architec.png)
