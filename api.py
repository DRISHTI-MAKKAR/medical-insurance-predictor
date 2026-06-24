# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd

app = FastAPI()
model         = joblib.load("insurance_xgb_model.pkl")
feature_names = joblib.load("feature_names.pkl")

class InsuranceInput(BaseModel):
    age: int
    sex: str        # "male" or "female"
    bmi: float
    children: int
    smoker: str     # "yes" or "no"
    region: str     # "northeast", "northwest", "southeast", "southwest"

def build_features(data):
    df = pd.DataFrame([data.dict()])
    def bmi_cat(b):
        if b < 18.5: return 'underweight'
        elif b < 25: return 'normal'
        elif b < 30: return 'overweight'
        else: return 'obese'
    df['bmi_category'] = df['bmi'].apply(bmi_cat)
    df['age_group']    = pd.cut(df['age'], bins=[17,30,45,60,100],
                                labels=['young','middle','senior','elderly'])
    df['smoker_obese'] = ((df['smoker']=='yes') & (df['bmi']>=30)).astype(int)
    df['bmi_squared']  = df['bmi'] ** 2
    df['age_squared']  = df['age'] ** 2
    df['sex']    = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df = pd.get_dummies(df, columns=['region','bmi_category','age_group'], drop_first=True)
    for col in feature_names:
        if col not in df.columns: df[col] = 0
    return df[feature_names].astype(float)

@app.post("/predict")
def predict(data: InsuranceInput):
    features   = build_features(data)
    log_pred   = model.predict(features)[0]
    prediction = max(1000.0, np.expm1(log_pred))
    return {
        "annual_cost":  round(float(prediction), 2),
        "monthly_cost": round(float(prediction / 12), 2),
        "range_low":    round(float(prediction * 0.85), 2),
        "range_high":   round(float(prediction * 1.15), 2)
    }