# AAPL Stock Price Prediction Using LSTM

This project demonstrates time-series forecasting with a Long Short-Term Memory
(LSTM) neural network. The model is trained on historical Apple Inc. (AAPL)
stock data and predicts the next closing price from the previous 60 trading
days.

The repository also includes a Streamlit app that loads the trained model and
scaler, displays the latest rows from the dataset, and generates a next-step
closing-price prediction.

> Disclaimer: This project is for educational machine learning practice only.
> It is not financial advice and should not be used for investment or trading
> decisions.

## Project Highlights

- Uses real historical AAPL stock data from 2006 to 2018.
- Builds a stacked LSTM model for one-step-ahead regression.
- Uses the previous 60 closing prices as the input sequence.
- Applies chronological train/test splitting for time-series integrity.
- Scales closing prices with `MinMaxScaler`.
- Evaluates predictions with MAE, MSE, and RMSE.
- Provides a simple Streamlit interface for inference.

## Dataset

The dataset is stored at:

```text
Dataset/AAPL_2006-01-01_to_2018-01-01.csv
```

It contains the following columns:

```text
Date, Open, High, Low, Close, Volume, Name
```

Only the `Close` column is used for model training and prediction. The dataset
is sorted by date, cleaned for missing values, and split chronologically into
training and testing sets.

## Problem Statement

This is a time-series regression problem:

```text
Previous 60 AAPL closing prices -> Next AAPL closing price
```

The model learns from historical closing-price sequences and estimates the next
value in the sequence.

## Workflow

```text
Load historical AAPL data
Clean and sort by date
Select the Close column
Split data chronologically into train and test sets
Scale prices with MinMaxScaler
Create 60-day input sequences
Train stacked LSTM model
Evaluate predictions
Save model and scaler
Deploy inference with Streamlit
```

## Model Architecture

The model is implemented with TensorFlow/Keras:

```text
Input shape: (60, 1)

LSTM(64, return_sequences=True)
LSTM(32)
Dense(1)
```

Training configuration:

```text
Optimizer: Adam
Loss: Mean Squared Error
Epochs: 30
Batch size: 32
Early stopping: Enabled on validation loss
```

The first LSTM layer returns sequences because it is followed by another LSTM
layer. The final dense layer outputs a single continuous value: the predicted
closing price.

## Evaluation

The trained model is evaluated on the test period using inverse-transformed
prices.

Results from the training notebook:

```text
MAE  : 2.9494
MSE  : 15.1717
RMSE : 3.8951
```

These values represent prediction error in the original stock-price scale.

## Project Structure

```text
Proj3/
|-- Dataset/
|   `-- AAPL_2006-01-01_to_2018-01-01.csv
|-- 1-StockPrediction.ipynb
|-- 2-Prediction.ipynb
|-- 3-app.py
|-- model.keras
|-- scaler.pkl
`-- README.md
```

### File Descriptions

- `1-StockPrediction.ipynb`: Loads data, preprocesses it, creates sequences,
  trains the LSTM model, evaluates performance, and saves the model artifacts.
- `2-Prediction.ipynb`: Loads the saved model and scaler, prepares the latest
  60-day sequence, and performs a sample prediction.
- `3-app.py`: Streamlit application for displaying dataset information and
  predicting the next AAPL closing price.
- `model.keras`: Saved trained LSTM model.
- `scaler.pkl`: Saved `MinMaxScaler` used during training and inference.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas scikit-learn tensorflow streamlit
```

If you are using Jupyter notebooks, also install:

```bash
pip install notebook matplotlib
```

## Running the App

From the project folder, run:

```bash
streamlit run 3-app.py
```

The app will open in your browser. Click the prediction button to generate the
next closing-price estimate from the latest 60 rows in the included dataset.

## Important Notes

- The app predicts the next value after the latest date available in the local
  dataset, not the current real-world AAPL price.
- The model uses only historical closing prices. It does not include market
  news, fundamentals, macroeconomic indicators, sentiment, or live market data.
- Stock prices are noisy and affected by many external factors, so this project
  should be treated as a learning exercise rather than a production forecasting
  system.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- Streamlit
- Jupyter Notebook

## Author

Repository owner: `heart-throbb`
