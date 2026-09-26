import streamlit as st
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.models import load_model

st.title("PM2.5 Air Quality Prediction")
st.write(
    "This application uses an LSTM neural "
    "network to predict the next-hour PM2.5 "
    "concentration."
)


@st.cache_resource
def load_lstm_model():
    return load_model("model.keras")


@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as file:
        return pickle.load(file)


model = load_lstm_model()
scaler = load_scaler()


@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/PRSA_Data_Guanyuan_20130301-20170228.csv")
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df[["datetime", "PM2.5"]]
    df = df.set_index("datetime")
    df = df.sort_index()
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df["PM2.5"] = df["PM2.5"].interpolate(method="time")
    df = df.dropna()
    return df


df = load_data()

SEQUENCE_LENGTH = 24
st.subheader("Dataset Information")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Latest PM2.5", f"{df['PM2.5'].iloc[-1]:.2f}")

st.subheader("Latest PM2.5 Measurements")
st.dataframe(df.tail(10), use_container_width=True)
st.subheader("LSTM Prediction")
if st.button("Predict Next Hour"):
    latest_data = df["PM2.5"].values[-SEQUENCE_LENGTH:]
    latest_scaled = scaler.transform(latest_data.reshape(-1, 1))
    X_input = latest_scaled.reshape(1, SEQUENCE_LENGTH, 1)
    prediction_scaled = model.predict(X_input, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)
    predicted_pm25 = prediction[0][0]
    st.success(f"Predicted Next-Hour PM2.5: " f"{predicted_pm25:.2f}")
