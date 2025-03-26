import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
dataset = load_diabetes()
df_diabetes = pd.DataFrame(dataset.data, columns=dataset.feature_names)
X = df_diabetes
y = dataset['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Streamlit UI
st.set_page_config(page_title="Smart Diabetes Prediction", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Risk Assessment", "🩺 Predict", "🤖 AI Health Guide"])

# Home Page
if page == "🏠 Home":
    st.title("💡 Smart Diabetes Prediction")
    st.markdown("An AI-driven tool for **predicting** and **assessing** diabetes risk.")
    st.image("https://source.unsplash.com/800x400/?health,diabetes", use_column_width=True)

# Risk Assessment
elif page == "📊 Risk Assessment":
    st.title("📊 Personalized Diabetes Risk Assessment")
    risk_factors = ["Body Mass Index (BMI)", "Age", "Blood Pressure Level"]
    user_risk = {factor: st.slider(f"{factor}", 1, 100, 50) for factor in risk_factors}
    risk_score = sum(user_risk.values()) / len(user_risk)
    st.metric(label="Estimated Diabetes Risk Score", value=f"{risk_score:.1f}")

# Prediction Page
elif page == "🩺 Predict":
    st.title("🩺 AI-Based Diabetes Prediction")
    user_input = {}
    feature_labels = {
        "age": "Age (Years)",
        "sex": "Gender (Male=1, Female=0)",
        "bmi": "Body Mass Index (BMI)",
        "bp": "Blood Pressure Level",
        "s1": "Total Cholesterol",
        "s2": "Low-Density Lipoproteins (LDL)",
        "s3": "High-Density Lipoproteins (HDL)",
        "s4": "Thyroid Stimulating Hormone",
        "s5": "Lamotrigine Levels",
        "s6": "Blood Sugar Level"
    }
    for feature, label in feature_labels.items():
        user_input[feature] = st.slider(f"{label}", float(X_train[feature].min()), float(X_train[feature].max()), float(X_train[feature].mean()))
    if st.button("Predict Diabetes Progression"):
        user_data = np.array([list(user_input.values())]).reshape(1, -1)
        prediction = model.predict(user_data)
        st.success(f"Predicted Diabetes Progression Score: {prediction[0]:.2f}")

# AI Health Guide
elif page == "🤖 AI Health Guide":
    st.title("🤖 Smart AI Health Assistant")
    user_query = st.text_input("Ask any diabetes-related question:")
    if user_query:
        response = "Based on AI analysis, maintaining a balanced diet and regular exercise is crucial for diabetes management. Always consult a healthcare professional for personalized advice."
        st.write(response)