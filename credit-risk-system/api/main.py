from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import create_model

app = FastAPI()


model = joblib.load('../models/model.joblib')
feature_columns = joblib.load('../models/feature_columns.joblib')

print("Model loaded.")


fields = {col: (float, 0.0) for col in feature_columns}
LoanApplication = create_model('LoanApplication', **fields)

@app.post('/predict')
def predict(loan: LoanApplication):
    data = pd.DataFrame([loan.dict()])
    data = data[feature_columns]
    probability = model.predict_proba(data)[0][1]
    label = "High Risk" if probability >= 0.5 else "Low Risk"

    return {"default_probability": round(float(probability), 4), "risk_label": label}