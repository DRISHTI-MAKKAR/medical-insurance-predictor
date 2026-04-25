import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model and feature names
model         = joblib.load('insurance_xgb_model.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥", layout="centered")

st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Fill in the details below to get an accurate estimate of your annual insurance cost.")
st.markdown("---")

# --- Input Form ---
col1, col2 = st.columns(2)

with col1:
    age      = st.slider("Age", 18, 64, 30)
    sex      = st.selectbox("Sex", ["male", "female"])
    bmi      = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                                help="Body Mass Index. Normal range: 18.5 - 24.9")

with col2:
    children = st.selectbox("Number of Children / Dependents", [0, 1, 2, 3, 4, 5])
    smoker   = st.selectbox("Smoker?", ["no", "yes"])
    region   = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# BMI reference guide
if bmi < 18.5:
    bmi_label = "Underweight"
elif bmi < 25:
    bmi_label = "Normal"
elif bmi < 30:
    bmi_label = "Overweight"
else:
    bmi_label = "Obese"
st.caption(f"BMI Category: {bmi_label}")

# --- Feature Engineering (matches training pipeline exactly) ---
def bmi_category(bmi):
    if bmi < 18.5:   return 'underweight'
    elif bmi < 25:   return 'normal'
    elif bmi < 30:   return 'overweight'
    else:            return 'obese'

def build_features(age, sex, bmi, children, smoker, region):
    d = {
        'age': age, 'sex': sex, 'bmi': bmi,
        'children': children, 'smoker': smoker, 'region': region
    }
    df = pd.DataFrame([d])
    df['bmi_category'] = df['bmi'].apply(bmi_category)
    df['age_group']    = pd.cut(df['age'], bins=[17, 30, 45, 60, 100],
                                labels=['young', 'middle', 'senior', 'elderly'])
    df['smoker_obese'] = ((df['smoker'] == 'yes') & (df['bmi'] >= 30)).astype(int)
    df['bmi_squared']  = df['bmi'] ** 2
    df['age_squared']  = df['age'] ** 2
    df['sex']    = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df = pd.get_dummies(df, columns=['region', 'bmi_category', 'age_group'], drop_first=True)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names].astype(float)

# --- Predict Button ---
st.markdown("---")
if st.button("Predict My Insurance Cost", type="primary", use_container_width=True):
    input_df = build_features(age, sex, bmi, children, smoker, region)

    # Predict in log space then inverse transform — mathematically guarantees positive output
    log_pred   = model.predict(input_df)[0]
    prediction = np.expm1(log_pred)
    prediction = max(1000.0, prediction)   # safety floor

    # Main result
    st.success(f"### Estimated Annual Cost: ${prediction:,.2f}")
    st.caption(f"Approximately ${prediction/12:,.2f} per month")

    # Confidence range
    low  = prediction * 0.85
    high = prediction * 1.15
    st.info(f"Likely range: ${low:,.2f} to ${high:,.2f}  (+-15% confidence band)")

    # Risk breakdown
    st.markdown("---")
    st.markdown("### Risk Factor Analysis")
    risk_score = 0

    if smoker == "yes":
        st.warning("Smoker — single biggest cost driver, adds $15,000-$25,000/year on average")
        risk_score += 3
    if bmi >= 30:
        st.warning("Obese (BMI >= 30) — increases insurance cost significantly")
        risk_score += 2
    if smoker == "yes" and bmi >= 30:
        st.error("Smoker + Obese — highest risk combination, expect significantly elevated premiums")
        risk_score += 2
    if age >= 50:
        st.warning("Age >= 50 — older age group incurs higher premiums")
        risk_score += 1
    if children >= 3:
        st.warning(f"{children} dependents — more dependents increase coverage cost")
        risk_score += 1
    if risk_score == 0:
        st.success("Low risk profile — non-smoker, healthy BMI, young age. You are in the best cost bracket!")

    st.markdown(f"**Overall Risk Score:** {risk_score}/7")
    st.progress(min(100, risk_score * 15))

    # Tips
    st.markdown("---")
    st.markdown("### How to Lower Your Premium")
    if smoker == "yes":
        st.markdown("- Quitting smoking is the single most impactful change — can save $15,000+ per year")
    if bmi >= 25:
        st.markdown(f"- Reducing BMI from {bmi:.1f} to below 25 (normal range) can meaningfully reduce cost")
    st.markdown("- Choosing a higher deductible plan can lower monthly premiums")
    st.markdown("- Region affects cost — northeast tends to be higher than southwest")
