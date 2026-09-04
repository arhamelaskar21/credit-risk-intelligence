import streamlit as st
import requests

st.title("RiskLens — Credit Risk Intelligence")
st.markdown("*Powered by XGBoost · Explainable AI · Real-time Risk Scoring*")
st.markdown("---")
st.markdown("### Loan Application Details")

col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0.0, value=10000.0)
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
    annual_inc = st.number_input("Annual Income ($)", min_value=0.0, value=60000.0)

with col2:
    term = st.selectbox("Loan Term", options=[0, 1], format_func=lambda x: "36 months" if x == 0 else "60 months")
    fico_range_low = st.number_input("FICO Score", min_value=300.0, max_value=850.0, value=680.0)
    loan_to_income = st.number_input("Loan to Income Ratio", min_value=0.0, value=0.17)

_, mid, _ = st.columns([1, 2, 1])
with mid:
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=15.0)

threshold = st.slider("Risk Threshold", min_value=0.1, max_value=0.9, value=0.5)
if st.button("Assess Risk"):
    payload = {
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "annual_inc": annual_inc,
        "dti": dti,
        "term": term,
        "fico_range_low": fico_range_low,
        "loan_to_income": loan_to_income
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    result = response.json()

    prob = result["default_probability"]
    label = "High Risk" if prob >= threshold else "Low Risk"

    if label == "High Risk":
        st.error(f"⚠️ {label} — Default Probability: {prob:.1%}")
    else:
        st.success(f"✅ {label} — Default Probability: {prob:.1%}")

st.markdown("---")
st.markdown("##### Model Info")
st.markdown("**Model:** XGBoost · **AUC:** 0.74 · **Recall (Default):** 0.69 · **Top Predictors:** Interest Rate, Term, DTI, Loan-to-Income")