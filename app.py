import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Fraud Detection 🚨")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # 🔥 KEEP ONLY 1 FEATURE (VERY IMPORTANT)
    # because scaler expects only 1 feature
    X = data[["Amount"]].values   # or "Time" depending on training

    # scale
    X_scaled = scaler.transform(X)

    # replace column
    data["Amount"] = X_scaled

    # drop non-model columns if needed
    model_input = data.select_dtypes(include=["number"])

    # predict
    prediction = model.predict(model_input)

    data["Prediction"] = prediction

    data["Prediction"] = data["Prediction"].map({
        0: "Legitimate",
        1: "Fraud"
    })

    st.subheader("Results")
    st.dataframe(data)