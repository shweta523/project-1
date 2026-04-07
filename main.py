import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Title
st.title("Diabetes Predictor (CSV Based)")

# Load CSV file
data = pd.read_csv("diabetes.csv")

# Show dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
st.subheader("Model Evaluation")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

# User input
st.subheader("Enter Patient Details")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    ]], columns=X.columns)

    result = model.predict(input_data)

    if result[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")
