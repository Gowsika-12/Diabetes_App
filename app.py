import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
bundle = joblib.load("diabetes_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("🩺 Diabetes Prediction App")
st.write(
    "Enter the patient measurements below to obtain "
    "an educational prediction from the trained model."
)

st.info(
    "For educational use only. This prediction is not "
    "a medical diagnosis."
)

# Patient input fields
pregnancies = st.number_input(
    "Pregnancies", min_value=0, max_value=20, value=0
)
glucose = st.number_input(
    "Glucose", min_value=0.0, max_value=300.0, value=120.0
)
blood_pressure = st.number_input(
    "Blood Pressure", min_value=0.0, max_value=200.0, value=70.0
)
skin_thickness = st.number_input(
    "Skin Thickness", min_value=0.0, max_value=100.0, value=20.0
)
insulin = st.number_input(
    "Insulin", min_value=0.0, max_value=900.0, value=80.0
)
bmi = st.number_input(
    "BMI", min_value=0.0, max_value=80.0, value=25.0
)
dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0, max_value=3.0, value=0.5
)
age = st.number_input(
    "Age", min_value=1, max_value=120, value=30
)

if st.button("Predict"):
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Apply the same scaling used during training
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning(
            "The model predicts a positive diabetes outcome."
        )
    else:
        st.success(
            "The model predicts a negative diabetes outcome."
        )

    st.caption(
        "This is a machine-learning prediction, not a diagnosis. "
        "Consult a qualified healthcare professional for assessment."
    )
