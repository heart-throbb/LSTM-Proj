import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.title("Household Energy Consumption Prediction")
st.write(
    "This application uses an LSTM neural network "
    "to predict the next 30-minute household "
    "electricity consumption."
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
    df = pd.read_csv(
        "Dataset/household_power_consumption.txt",
        sep=";",
        na_values="?",
        low_memory=False,
    )
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"], errors="coerce"
    )
    df = df[["Datetime", "Global_active_power"]]
    df = df.set_index("Datetime")
    df = df.sort_index()
    df["Global_active_power"] = df["Global_active_power"].interpolate(method="time")
    df = df.dropna()
    df = df.resample("30min").mean()
    df = df.dropna()
    return df


df = load_data()

SEQUENCE_LENGTH = 60

st.subheader("Dataset Information")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Latest Power", f"{df['Global_active_power'].iloc[-1]:.3f} kW")

st.subheader("Latest Measurements")
st.dataframe(df.tail(10), use_container_width=True)
st.subheader("LSTM Prediction")
if st.button("Predict Next 30-Minute Consumption"):
    latest_data = df["Global_active_power"].values[-SEQUENCE_LENGTH:]
    latest_scaled = scaler.transform(latest_data.reshape(-1, 1))
    X_input = latest_scaled.reshape(1, SEQUENCE_LENGTH, 1)
    prediction_scaled = model.predict(X_input, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)
    predicted_power = prediction[0][0]
    st.success(f"Predicted Power Consumption: " f"{predicted_power:.4f} kW")
