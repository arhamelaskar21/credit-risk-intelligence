# RiskLens — Credit Risk Intelligence System

RiskLens is an end-to-end credit risk intelligence system that predicts the probability of loan default. The system is trained on 44,000+ real loan applications. It uses XGBoost as the model. It includes SHAP explainability to satisfy regulatory requirements for adverse action notices. The system is served via a FastAPI endpoint for real-time scoring and a Streamlit dashboard for loan officer decision support.

## Tech Stack
- **Model:** XGBoost (AUC 0.74, Default Recall 0.69)
- **Explainability:** SHAP TreeExplainer
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit
- **Data:** 44,252 loans, 76 features, 20.96% default rate

## Project Structure
credit-risk-system/
data/ ← raw and processed data
notebooks/ ← EDA and modeling
api/ ← FastAPI app
streamlit_app/ ← Streamlit dashboard
models/ ← saved model artifacts
DECISIONS.md ← key technical decisions


## How It Works
1. Loan application data is sent via POST request to `/predict`
2. FastAPI validates and passes data to the XGBoost model
3. Model returns default probability + risk label
4. Streamlit dashboard displays result with adjustable risk threshold

## Key Decisions
- XGBoost selected over Logistic Regression for higher default recall (0.69 vs 0.66)
- SHAP used for regulatory compliance and explainability
- Threshold adjustable by loan officer — business decision, not hardcoded

## Results
| Model | AUC | Default Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.74 | 0.66 | 0.47 |
| XGBoost (tuned) | 0.74 | 0.69 | 0.46 |