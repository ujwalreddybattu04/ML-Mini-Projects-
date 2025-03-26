Smart Diabetes Prediction UI
An AI-powered tool to predict diabetes progression and assess personalized risk using machine learning and Streamlit.

🚀 Features
✅ Diabetes Prediction using ML models
✅ Personalized Risk Assessment with interactive sliders
✅ AI Health Guide for diabetes-related queries
✅ User-friendly UI built with Streamlit

🧪 Tech Stack
Python (Core programming language)
Streamlit (Web UI)
Scikit-Learn (Machine Learning)
Pandas & NumPy (Data Processing)
Pickle (Model Storage)

 Install Dependencies
 pip install -r requirements.txt

Run the Streamlit App
streamlit run app.py

Dataset Information
The model is trained on the Diabetes dataset from sklearn.datasets.load_diabetes(). The dataset contains:
Independent Variables (Features): Age, BMI, Blood Pressure, Cholesterol, Lipoproteins, etc.
Dependent Variable (Target): Diabetes progression measurement

🖥️ Usage
1️⃣ Navigate through different sections:
🏠 Home → Overview of the app
📊 Risk Assessment → Get a personal diabetes risk score
🩺 Predict → Predict diabetes progression using ML
🤖 AI Health Guide → Get AI-driven health insights
2️⃣ Provide Inputs using sliders for BMI, age, blood pressure, etc.
3️⃣ Get Predictions based on a pre-trained RandomForestRegressor model.

📌 Future Improvements
🔹 Integration with real-time health data
🔹 More advanced ML models (Deep Learning, XGBoost)
🔹 Chatbot for personalized diabetes guidance