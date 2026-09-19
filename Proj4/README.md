# Household Energy Consumption Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict the next 30-minute household electricity consumption value from previous power-consumption readings.

The project is based on the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

## Project Overview

The model uses historical `Global_active_power` values to forecast the next electricity consumption value.

The original dataset contains one-minute household electricity measurements collected from a house in Sceaux, France, between December 2006 and November 2010. In this project, the data is resampled into 30-minute intervals and then used for time-series forecasting.

## Dataset

Dataset: Individual Household Electric Power Consumption

Source: UCI Machine Learning Repository

Dataset page: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

DOI: `10.24432/C58K54`

The original dataset contains:

- Date
- Time
- Global active power
- Global reactive power
- Voltage
- Global intensity
- Sub metering 1
- Sub metering 2
- Sub metering 3

This implementation uses only:

```text
Global_active_power
```

The raw extracted dataset file is about 127 MB, so it should not be committed directly to GitHub. Download the dataset from UCI, extract it, and place the file here:

```text
Dataset/household_power_consumption.txt
```

## Problem Statement

The goal is to predict the next 30-minute household electricity consumption value using the previous 60 observations.

Since each observation represents 30 minutes:

```text
60 observations x 30 minutes = 30 hours of previous data
```

Input:

```text
Previous 60 Global_active_power values
```

Output:

```text
Next 30-minute Global_active_power value in kW
```

## Project Pipeline

```text
Load dataset
Combine Date and Time columns
Convert Global_active_power to numeric values
Handle missing values
Interpolate missing time-series values
Resample data into 30-minute intervals
Create chronological train-test split
Scale data using MinMaxScaler
Create 60-step input sequences
Train LSTM model
Evaluate model
Save model and scaler
Run prediction notebook or Streamlit app
```

## Data Preprocessing

The preprocessing steps implemented in the notebook and app are:

1. Load `household_power_consumption.txt`.
2. Treat `?` values as missing values.
3. Combine `Date` and `Time` into a single datetime column.
4. Convert `Global_active_power` to numeric.
5. Keep only `Datetime` and `Global_active_power`.
6. Sort the data by datetime.
7. Interpolate missing values using time-based interpolation.
8. Resample readings to 30-minute intervals.
9. Split the data chronologically into training and testing sets.
10. Scale values with `MinMaxScaler`.
11. Create 60-step sequences for LSTM training.

## Model Architecture

The implemented model is a stacked LSTM regression model:

```python
model = Sequential(
    [
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
        LSTM(32),
        Dense(1),
    ]
)
```

The model is compiled with:

```python
model.compile(optimizer="adam", loss="mean_squared_error")
```

## Training Details

- Sequence length: `60`
- Train/test split: `80% / 20%`
- Epochs: `30`
- Batch size: `64`
- Validation split: `0.1`
- Optimizer: Adam
- Loss function: Mean Squared Error
- Callback: Early stopping

Early stopping is configured as:

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

In the saved notebook run, training stopped after 14 epochs.

## Evaluation

The model is evaluated using:

- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error

Saved notebook results:

```text
Test loss: 0.003999
MAE: 0.3256 kW
MSE: 0.2426
RMSE: 0.4925 kW
```

These values can change if the model is retrained.

## Project Structure

```text
Proj4/
|-- Dataset/
|   `-- household_power_consumption.txt
|-- 1_EnergyPrediction.ipynb
|-- 2-Prediction.ipynb
|-- 3-app.py
|-- model.keras
|-- scaler.pkl
`-- README.md
```

## Requirements

Install the required packages:

```bash
pip install tensorflow pandas numpy scikit-learn streamlit
```

Required Python packages:

```text
tensorflow
pandas
numpy
scikit-learn
streamlit
```

## How to Run

First, download the dataset from UCI and place `household_power_consumption.txt` inside the `Dataset/` folder.

Run the training notebook:

```text
1_EnergyPrediction.ipynb
```

This notebook trains the LSTM model and creates:

```text
model.keras
scaler.pkl
```

Run the prediction notebook:

```text
2-Prediction.ipynb
```

This notebook loads the saved model and scaler, prepares the latest 60 observations, and predicts the next 30-minute consumption value.

Run the Streamlit app:

```bash
streamlit run 3-app.py
```

The Streamlit app loads the trained model, loads the saved scaler, preprocesses the dataset, displays recent measurements, and predicts the next 30-minute household power consumption.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- LSTM neural networks
- Streamlit
- Jupyter Notebook

## Dataset Citation

Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption. UCI Machine Learning Repository. DOI: `10.24432/C58K54`

## Author

Repository owner: `heart-throbb`
