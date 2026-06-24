import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# Step 1: Load dataset
# -------------------------------------------------------
df = pd.read_csv('insurance.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------------------------------
# Step 2: Feature Engineering
# -------------------------------------------------------
def bmi_category(bmi):
    if bmi < 18.5:   return 'underweight'
    elif bmi < 25:   return 'normal'
    elif bmi < 30:   return 'overweight'
    else:            return 'obese'

df['bmi_category'] = df['bmi'].apply(bmi_category)

df['age_group'] = pd.cut(
    df['age'],
    bins=[17, 30, 45, 60, 100],
    labels=['young', 'middle', 'senior', 'elderly']
)

df['smoker_obese'] = ((df['smoker'] == 'yes') & (df['bmi'] >= 30)).astype(int)
df['bmi_squared']  = df['bmi'] ** 2
df['age_squared']  = df['age'] ** 2

df['sex']    = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

df = pd.get_dummies(df, columns=['region', 'bmi_category', 'age_group'], drop_first=True)

# -------------------------------------------------------
# Step 3: Target variable (log transform)
# -------------------------------------------------------
df['log_charges'] = np.log1p(df['charges'])

# -------------------------------------------------------
# Step 4: Define features
# -------------------------------------------------------
feature_names = [col for col in df.columns if col not in ['charges', 'log_charges']]
X = df[feature_names]
y = df['log_charges']

print(f"Features: {feature_names}")

# -------------------------------------------------------
# Step 5: Train/Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# -------------------------------------------------------
# Step 6: Train XGBoost model
# -------------------------------------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)
print("Model trained successfully!")

# -------------------------------------------------------
# Step 7: Check accuracy
# -------------------------------------------------------
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model R2 Score: {r2:.4f}")

# -------------------------------------------------------
# Step 8: Save model and feature names
# -------------------------------------------------------
joblib.dump(model,         'insurance_xgb_model.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
print("\n✅ Models saved successfully!")
print("   - insurance_xgb_model.pkl")
print("   - feature_names.pkl")
print("\nNow push these files to GitHub!")