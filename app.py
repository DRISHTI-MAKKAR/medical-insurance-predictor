import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from groq import Groq
from dotenv import load_dotenv
import os


# -------------------------------------------------------
# Load ML model
# -------------------------------------------------------
model         = joblib.load('insurance_xgb_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# -------------------------------------------------------
# Groq AI client
# -------------------------------------------------------
load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥", layout="wide")

st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Predict your insurance cost using the form **or** ask our AI assistant directly!")
st.markdown("---")

# -------------------------------------------------------
# Feature Engineering (same as training pipeline)
# -------------------------------------------------------
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

def predict_cost(age, sex, bmi, children, smoker, region):
    input_df   = build_features(age, sex, bmi, children, smoker, region)
    log_pred   = model.predict(input_df)[0]
    prediction = np.expm1(log_pred)
    prediction = max(1000.0, prediction)
    return prediction

# -------------------------------------------------------
# Groq AI tool definition — ML model as a tool
# -------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "predict_insurance_cost",
            "description": "Predicts annual medical insurance cost based on patient details",
            "parameters": {
                "type": "object",
                "properties": {
                    "age":      {"type": "integer", "description": "Age of the person (18 to 64)"},
                    "sex":      {"type": "string",  "description": "Gender: male or female"},
                    "bmi":      {"type": "number",  "description": "Body Mass Index (e.g. 25.0)"},
                    "children": {"type": "integer", "description": "Number of children (0 to 5)"},
                    "smoker":   {"type": "string",  "description": "Smoker: yes or no"},
                    "region":   {"type": "string",  "description": "US Region: northeast, northwest, southeast, southwest"}
                },
                "required": ["age", "sex", "bmi", "children", "smoker", "region"]
            }
        }
    }
]

# -------------------------------------------------------
# Ask Groq AI — with ML model tool calling
# -------------------------------------------------------
def ask_groq_ai(user_question, chat_history):
    # Build messages from chat history
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly medical insurance advisor. "
                "When users ask about insurance costs, always use the predict_insurance_cost tool. "
                "If the user has not provided all required details (age, sex, bmi, children, smoker, region), "
                "make reasonable assumptions and mention what you assumed. "
                "Explain results clearly with tips to reduce costs."
            )
        }
    ]

    # Add chat history
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current question
    messages.append({"role": "user", "content": user_question})

    # First call — Groq decides whether to call ML model
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1024
    )

    response_message = response.choices[0].message

    # If Groq calls your ML model
    if response_message.tool_calls:
        tool_call     = response_message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)

        # Run your XGBoost ML model
        prediction = predict_cost(
            age      = function_args.get("age", 30),
            sex      = function_args.get("sex", "male"),
            bmi      = function_args.get("bmi", 25.0),
            children = function_args.get("children", 0),
            smoker   = function_args.get("smoker", "no"),
            region   = function_args.get("region", "northeast")
        )

        ml_result = {
            "annual_cost":  round(float(prediction), 2),
            "monthly_cost": round(float(prediction / 12), 2),
            "range_low":    round(float(prediction * 0.85), 2),
            "range_high":   round(float(prediction * 1.15), 2)
        }

        # Send ML result back to Groq for human-friendly explanation
        messages.append({"role": "assistant", "content": "", "tool_calls": response_message.tool_calls})
        messages.append({
            "role":        "tool",
            "tool_call_id": tool_call.id,
            "name":        tool_call.function.name,
            "content":     json.dumps(ml_result)
        })

        final_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=1024
        )

        return final_response.choices[0].message.content

    return response_message.content

# -------------------------------------------------------
# TWO COLUMN LAYOUT
# Left = original form | Right = AI chat
# -------------------------------------------------------
col_form, col_chat = st.columns([1, 1])

# ==================== LEFT: ORIGINAL FORM ====================
with col_form:
    st.subheader("📋 Predict Using Form")

    c1, c2 = st.columns(2)
    with c1:
        age      = st.slider("Age", 18, 64, 30)
        sex      = st.selectbox("Sex", ["male", "female"])
        bmi      = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                                    help="Body Mass Index. Normal range: 18.5 - 24.9")
    with c2:
        children = st.selectbox("Number of Children / Dependents", [0, 1, 2, 3, 4, 5])
        smoker   = st.selectbox("Smoker?", ["no", "yes"])
        region   = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    # BMI label
    if bmi < 18.5:       bmi_label = "Underweight"
    elif bmi < 25:       bmi_label = "Normal"
    elif bmi < 30:       bmi_label = "Overweight"
    else:                bmi_label = "Obese"
    st.caption(f"BMI Category: {bmi_label}")

    st.markdown("---")
    if st.button("Predict My Insurance Cost", type="primary", use_container_width=True):
        prediction = predict_cost(age, sex, bmi, children, smoker, region)
        low        = prediction * 0.85
        high       = prediction * 1.15

        st.success(f"### Estimated Annual Cost: ${prediction:,.2f}")
        st.caption(f"Approximately ${prediction/12:,.2f} per month")
        st.info(f"Likely range: ${low:,.2f} to ${high:,.2f}  (±15% confidence band)")

        st.markdown("---")
        st.markdown("### Risk Factor Analysis")
        risk_score = 0
        if smoker == "yes":
            st.warning("Smoker — biggest cost driver, adds $15,000-$25,000/year")
            risk_score += 3
        if bmi >= 30:
            st.warning("Obese (BMI >= 30) — increases cost significantly")
            risk_score += 2
        if smoker == "yes" and bmi >= 30:
            st.error("Smoker + Obese — highest risk combination")
            risk_score += 2
        if age >= 50:
            st.warning("Age >= 50 — older age group incurs higher premiums")
            risk_score += 1
        if children >= 3:
            st.warning(f"{children} dependents — more dependents increase cost")
            risk_score += 1
        if risk_score == 0:
            st.success("Low risk profile — you are in the best cost bracket!")

        st.markdown(f"**Overall Risk Score:** {risk_score}/7")
        st.progress(min(100, risk_score * 15))

        st.markdown("---")
        st.markdown("### How to Lower Your Premium")
        if smoker == "yes":
            st.markdown("- Quitting smoking can save $15,000+ per year")
        if bmi >= 25:
            st.markdown(f"- Reducing BMI from {bmi:.1f} to below 25 can lower cost")
        st.markdown("- Choosing a higher deductible plan lowers monthly premiums")
        st.markdown("- Northeast region tends to be higher than southwest")

# ==================== RIGHT: AI CHAT ====================
with col_chat:
    st.subheader("🤖 Ask AI Assistant")
    st.caption("Ask anything in plain English — no need to fill the form!")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    chat_container = st.container(height=450)
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Hi! Ask me about your insurance cost. Example: 'I am 28 years old female, non-smoker, BMI 24, no kids, from northeast. What will my cost be?'")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])

    # Chat input box
    user_input = st.chat_input("Ask about your insurance cost...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get AI response
        with st.spinner("AI is thinking..."):
            try:
                ai_response = ask_groq_ai(user_input, st.session_state.chat_history[:-1])
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}"
                })

        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()