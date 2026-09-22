import streamlit as st
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.models import load_model

st.title("Traffic Volume Prediction")
st.write(
    "This application uses an LSTM neural "
    "network to predict the next hour's "
    "traffic volume."
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
    df = pd.read_csv("Dataset/" "Metro_Interstate_Traffic_Volume.csv")
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time")
    df = df[["date_time", "traffic_volume"]]
    df = df.set_index("date_time")
    return df


df = load_data()
SEQUENCE_LENGTH = 24
st.subheader("Dataset Information")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Latest Traffic", f"{df['traffic_volume'].iloc[-1]:.0f}")

st.subheader("Latest Traffic Measurements")
st.dataframe(df.tail(10), use_container_width=True)
st.subheader("LSTM Prediction")
if st.button("Predict Next Hour"):
    latest_data = df["traffic_volume"].values[-SEQUENCE_LENGTH:]
    latest_scaled = scaler.transform(latest_data.reshape(-1, 1))
    X_input = latest_scaled.reshape(1, SEQUENCE_LENGTH, 1)
    prediction_scaled = model.predict(X_input, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)
    predicted_traffic = prediction[0][0]
    st.success(f"Predicted Traffic Volume: " f"{predicted_traffic:.0f} vehicles/hour")
