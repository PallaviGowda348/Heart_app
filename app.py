import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("❤️ Heart Attack Risk Prediction")

# Inputs
age = st.slider("Age", 20, 100, 30)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
restingBP = st.number_input("Resting BP", min_value=80, value=120)
chol = st.number_input("Cholesterol", min_value=100, value=200)
fbs = st.selectbox("Fasting Blood Sugar >120", [0, 1])
restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
maxhr = st.number_input("Max Heart Rate", min_value=60, value=150)
exang = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", min_value=0.0, value=1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Encoding
sex = 1 if sex == "M" else 0

cp_dict = {"ATA":0, "NAP":1, "ASY":2, "TA":3}
cp = cp_dict[cp]

restecg_dict = {"Normal":0, "ST":1, "LVH":2}
restecg = restecg_dict[restecg]

exang = 1 if exang == "Y" else 0

slope_dict = {"Up":0, "Flat":1, "Down":2}
slope = slope_dict[slope]

# Predict
if st.button("Predict Risk"):
    data = np.array([[age, sex, cp, restingBP, chol,
                      fbs, restecg, maxhr, exang,
                      oldpeak, slope]])

    data = scaler.transform(data)

    prob = model.predict_proba(data)
    result = model.predict(data)

    st.write(f"🟢 Low Risk: {prob[0][0]*100:.2f}%")
    st.write(f"🔴 High Risk: {prob[0][1]*100:.2f}%")

    if result[0] == 1:
        st.error("⚠ HIGH RISK of Heart Attack")
    else:
        st.success("✅ LOW RISK of Heart Attack")