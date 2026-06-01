import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('final_model.pkl')
scaler = joblib.load('scaler.pkl')
st.sidebar.header("About")
st.sidebar.write(
    "This app predicts the likelihood of heart disease using a tuned Random Forest Machine Learning model."
)
st.title("💖 Heart Disease Prediction App")
st.subheader("Machine Learning Based Heart Disease Prediction")
st.write("Enter patient details below:")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=45,
    step=1,
    format="%d"
)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

trestbps = st.number_input("Resting Blood Pressure")
trestbps = st.number_input(
    "Resting blood pressure",
    min_value=0,
    max_value=300,
    value=120,
    step=1,
    format="%d"
)
chol = st.number_input(
    "Cholesterol",
    min_value=0,
    max_value=600,
    value=200,
    step=1,
    format="%d"
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

thalch = st.number_input(
    "Maximum Heart Rate",
    min_value=0,
    max_value=600,
    value=150,
    step=1,
    format="%d"
)
exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input("Oldpeak")

cp = st.selectbox(
    "Chest Pain Type",
    [
        "atypical angina",
        "non-anginal",
        "typical angina",
        "asymptomatic"
    ]
)

restecg = st.selectbox(
    "Rest ECG",
    [
        "normal",
        "st-t abnormality",
        "left ventricular hypertrophy"
    ]
)

slope = st.selectbox(
    "Slope",
    [
        "flat",
        "upsloping",
        "downsloping"
    ]
)

# Encoding
sex = 1 if sex == "Male" else 0

cp_atypical = 1 if cp == "atypical angina" else 0
cp_non_anginal = 1 if cp == "non-anginal" else 0
cp_typical = 1 if cp == "typical angina" else 0

restecg_normal = 1 if restecg == "normal" else 0
restecg_st = 1 if restecg == "st-t abnormality" else 0

slope_flat = 1 if slope == "flat" else 0
slope_upsloping = 1 if slope == "upsloping" else 0

# Create dataframe
input_data = pd.DataFrame([{

    'age': age,
    'sex': sex,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'thalch': thalch,
    'exang': exang,
    'oldpeak': oldpeak,

    'cp_atypical angina': cp_atypical,
    'cp_non-anginal': cp_non_anginal,
    'cp_typical angina': cp_typical,

    'restecg_normal': restecg_normal,
    'restecg_st-t abnormality': restecg_st,

    'slope_flat': slope_flat,
    'slope_upsloping': slope_upsloping

}])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)

probability = model.predict_proba(input_scaled)

# Button
if st.button("Predict"):

    if prediction[0] == 1:

        st.error("⚠️ High Risk of Heart Disease")

    else:

        st.success("✅ Low Risk of Heart Disease")

    disease_prob = probability[0][1] * 100

    st.write(f"Heart Disease Probability: {disease_prob:.2f}%")
st.progress(int(disease_prob))
st.markdown("---")

st.write(
    "This project uses a tuned Random Forest Classifier to predict the likelihood of heart disease."
)
st.sidebar.write("Model Accuracy: 82.39%")
st.sidebar.write("ROC-AUC Score: 0.90")
st.markdown("---")
with st.expander("Feature Information"):
    st.write("""
    - Oldpeak: ST depression induced by exercise
    """)
st.caption(
    "Disclaimer: This prediction is based on a machine learning model and should not replace professional medical advice."
)
