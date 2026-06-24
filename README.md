# 🏥 Medical Insurance Cost Predictor

> **Live App:** [https://drishti-makkar-medical-insurance-predictor-app-nzjqa7.streamlit.app/](https://drishti-makkar-medical-insurance-predictor-app-nzjqa7.streamlit.app/)

An AI-powered medical insurance cost prediction app built with **XGBoost ML model** and **Groq AI (LLaMA 3.1)**, wrapped in a **Streamlit** web interface. Users can either fill a form or chat with the AI assistant in plain English to get their insurance cost estimate.

---

## 🚀 Live Demo

🔗 **[Click here to try the app](https://drishti-makkar-medical-insurance-predictor-app-nzjqa7.streamlit.app/)**

---

## ✨ Features

- 📋 **Form-based prediction** — Fill in your details and get instant cost estimate
- 🤖 **AI Chat Assistant** — Ask in plain English, Groq AI calls the ML model and explains results
- 📊 **Risk Factor Analysis** — See what factors are driving your insurance cost
- 💡 **Cost Reduction Tips** — Get personalized tips to lower your premium
- 📈 **Confidence Range** — See ±15% cost range for better planning

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost (trained on insurance dataset) |
| AI Integration | Groq AI — LLaMA 3.1 8B (free) |
| Frontend | Streamlit |
| Model Serving | FastAPI + Uvicorn |
| Language | Python 3.13 |
| Deployment | Streamlit Cloud |

---

## 🔁 How AI + ML Are Connected

```
User types question in plain English
            ↓
Groq AI reads and understands the question
            ↓
Groq extracts: age, sex, bmi, children, smoker, region
            ↓
Calls XGBoost ML Model via predict_cost function
            ↓
ML Model returns: annual cost, monthly cost, range
            ↓
Groq explains result in human-friendly language
            ↓
User sees full answer with tips and risk analysis
```

> **Important:** The numbers always come from the XGBoost ML model. Groq AI only understands the question and explains the result — it never calculates the cost itself.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Medical Insurance Dataset |
| Rows | 1338 records |
| Features | age, sex, bmi, children, smoker, region |
| Target | charges (annual insurance cost in USD) |
| Model | XGBoost Regressor |
| R2 Score | ~0.89 |

---

## 📁 Project Structure

```
medical_insurance_project/
├── app.py                        # Streamlit web app with AI chat
├── api.py                        # FastAPI ML model server
├── grok_agent.py                 # Groq AI agent (terminal version)
├── retrain.py                    # Script to retrain model
├── insurance.csv                 # Training dataset
├── insurance_xgb_model.pkl       # Trained XGBoost model
├── feature_names.pkl             # Feature names list
├── requirements.txt              # Python dependencies
├── .env                          # API keys (never pushed to GitHub)
├── .gitignore                    # Ignores .env and cache files
└── README.md                     # This file
```

---

## ⚙️ Local Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/DRISHTI-MAKKAR/medical-insurance-predictor.git
cd medical-insurance-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env` file
Create a file called `.env` in the project folder:
```
GROQ_API_KEY=gsk_your_groq_api_key_here
```
Get your **free** Groq API key from: [https://console.groq.com/keys](https://console.groq.com/keys)

### 4. Run the app

Open **two terminals:**

**Terminal 1 — Start ML model server:**
```bash
uvicorn api:app --reload --port 8000
```

**Terminal 2 — Start Streamlit app:**
```bash
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 💬 How to Use

### Option 1 — Use the Form (Left side)
1. Fill in age, sex, BMI, number of children, smoker status, region
2. Click **Predict My Insurance Cost**
3. See predicted cost, risk analysis, and tips

### Option 2 — Chat with AI (Right side)
Type naturally in the chat box:
```
"I am 28 year old female, BMI 24, non-smoker, 0 kids, northeast. What is my cost?"
```
```
"I am 45 male, smoker, BMI 32, 2 children, southeast region"
```
The AI will automatically extract your details, call the ML model, and explain your insurance cost with personalized tips.

---

## 📦 Requirements

```
streamlit
pandas
numpy
xgboost==3.0.5
scikit-learn
joblib==1.5.2
groq
requests
python-dotenv
```

---

## 🔐 Security

- API keys stored in `.env` file only — never pushed to GitHub
- `.env` is listed in `.gitignore`
- Anyone cloning this repo must create their own `.env` with their own Groq API key
- Streamlit Cloud uses Secrets management for deployment

---

## 🧠 ML Model Details

The XGBoost model was trained with the following engineered features:

| Feature | Description |
|---|---|
| age, sex, bmi, children, smoker, region | Raw input features |
| bmi_squared | BMI squared for non-linear effects |
| age_squared | Age squared for non-linear effects |
| smoker_obese | Combined smoker + obese risk flag |
| bmi_category | underweight / normal / overweight / obese |
| age_group | young / middle / senior / elderly |
| region dummies | One-hot encoded region columns |

Target variable was **log-transformed** (`np.log1p`) during training and inverse-transformed (`np.expm1`) during prediction for better accuracy.

---

## 👩‍💻 Author

**Drishti Makkar**
GitHub: [@DRISHTI-MAKKAR](https://github.com/DRISHTI-MAKKAR)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
