# grok_agent.py
# Medical Insurance Cost Predictor using Groq AI (Free)

from groq import Groq
import requests
import json
from dotenv import load_dotenv
import os

# -------------------------------------------------------
# Step 1: Connect to Groq (Free AI)
# Get your free API key from https://console.groq.com
# -------------------------------------------------------
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------------------------------
# Step 1: Connect to Groq (Free AI)
# Get your free API key from https://console.groq.com
# -------------------------------------------------------

# -------------------------------------------------------
# Step 2: Define your ML model as a tool for Groq AI
# Groq will automatically decide when to call this tool
# -------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "predict_insurance_cost",
            "description": "Predicts annual medical insurance cost based on patient details like age, sex, BMI, number of children, smoking status, and region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "description": "Age of the person (between 18 and 64)"
                    },
                    "sex": {
                        "type": "string",
                        "description": "Gender of the person. Must be 'male' or 'female'"
                    },
                    "bmi": {
                        "type": "number",
                        "description": "Body Mass Index of the person (e.g. 28.5)"
                    },
                    "children": {
                        "type": "integer",
                        "description": "Number of children or dependents (0 to 5)"
                    },
                    "smoker": {
                        "type": "string",
                        "description": "Whether the person smokes. Must be 'yes' or 'no'"
                    },
                    "region": {
                        "type": "string",
                        "description": "Region in the US. Must be one of: northeast, northwest, southeast, southwest"
                    }
                },
                "required": ["age", "sex", "bmi", "children", "smoker", "region"]
            }
        }
    }
]

# -------------------------------------------------------
# Step 3: Function to call your ML model API (api.py)
# Make sure api.py is running on port 8000 first!
# -------------------------------------------------------
def call_ml_model(inputs):
    """Sends inputs to your FastAPI model server and returns prediction"""
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=inputs,
            timeout=10
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "ML model server is not running. Please start api.py first using: uvicorn api:app --reload --port 8000"}

# -------------------------------------------------------
# Step 4: Main function — ask Groq a question
# -------------------------------------------------------
def ask_groq(user_question):
    """
    Send a natural language question to Groq AI.
    Groq will automatically call your ML model and explain the result.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and friendly medical insurance advisor. "
                "When a user asks about insurance costs, use the predict_insurance_cost tool "
                "to get the prediction. Then explain the result clearly in simple language. "
                "Also mention what factors are driving the cost and give 2-3 tips to reduce it."
            )
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    # --- First call: Groq reads question and decides to call the tool ---
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # free model on Groq
        messages=messages,
        tools=tools,
        tool_choice="auto",        # Groq decides when to call your ML model
        max_tokens=1024
    )

    response_message = response.choices[0].message

    # --- If Groq decided to call your ML model ---
    if response_message.tool_calls:
        tool_call     = response_message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)

        print("\n--- Groq extracted these values from your question ---")
        print(f"  Age:      {function_args.get('age')}")
        print(f"  Sex:      {function_args.get('sex')}")
        print(f"  BMI:      {function_args.get('bmi')}")
        print(f"  Children: {function_args.get('children')}")
        print(f"  Smoker:   {function_args.get('smoker')}")
        print(f"  Region:   {function_args.get('region')}")

        # --- Call your XGBoost ML model ---
        print("\n--- Calling your ML model... ---")
        ml_result = call_ml_model(function_args)
        print(f"  ML Model Result: {ml_result}")

        # --- Send ML result back to Groq for a human-friendly explanation ---
        messages.append({"role": "assistant", "content": "", "tool_calls": response_message.tool_calls})
        messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": json.dumps(ml_result)
})

        final_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=1024
        )

        return final_response.choices[0].message.content

    # --- If Groq answered directly without calling the tool ---
    return response_message.content


# -------------------------------------------------------
# Step 5: Test it with different questions
# -------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("   Medical Insurance Cost Predictor — Powered by Groq AI")
    print("=" * 60)

    # Test question 1
    question1 = "I am a 35 year old male, BMI 29, non-smoker, 2 kids, from southeast. What will my insurance cost?"
    print(f"\nQuestion: {question1}")
    print("\nGroq Answer:", ask_groq(question1))

    print("\n" + "=" * 60)

    # Test question 2
    question2 = "I am a 45 year old female, BMI 32, I smoke, 1 child, living in northeast. How much will I pay?"
    print(f"\nQuestion: {question2}")
    print("\nGroq Answer:", ask_groq(question2))

    print("\n" + "=" * 60)

    # Interactive mode — ask your own question
    print("\n--- Ask your own question ---")
    user_input = input("Your question: ")
    if user_input.strip():
        print("\nGroq Answer:", ask_groq(user_input))