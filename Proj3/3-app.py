import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.title("AAPL Stock Price Prediction using LSTM")
st.write(
    "This application uses an LSTM neural network "
    "to predict the next AAPL closing price based "
    "on the previous 60 trading days."
)


@st.cache_resource
def load_lstm_model():
    return load_model("model.keras")


@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/AAPL_2006-01-01_to_2018-01-01.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


model = load_lstm_model()
scaler = load_scaler()
df = load_data()

st.subheader("Dataset Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Stock", "AAPL")
with col2:
    st.metric("Rows", len(df))
with col3:
    st.metric("Sequence Length", "60 Days")

st.subheader("Latest Historical Data")
st.dataframe(df.tail(10), use_container_width=True)
st.subheader("Prediction")

if st.button("Predict Next Closing Price"):
    SEQUENCE_LENGTH = 60
    close_prices = df[["Close"]].values
    scaled_prices = scaler.transform(close_prices)
    last_60_days = scaled_prices[-SEQUENCE_LENGTH:]
    X_input = np.array([last_60_days])
    prediction_scaled = model.predict(X_input, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)
    predicted_price = prediction[0][0]
    latest_price = df["Close"].iloc[-1]
    st.success(f"Predicted Next AAPL Closing Price: " f"${predicted_price:.2f}")
    st.info(f"Latest Available AAPL Closing Price: " f"${latest_price:.2f}")
