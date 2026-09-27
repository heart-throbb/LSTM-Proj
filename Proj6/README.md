# PM2.5 Air Quality Prediction Using LSTM

A time-series forecasting project that predicts the next-hour PM2.5 concentration at Beijing’s Guanyuan monitoring station using a Long Short-Term Memory (LSTM) neural network.

## Overview

PM2.5 is fine particulate matter with an aerodynamic diameter of 2.5 micrometers or less. Its concentration is measured in micrograms per cubic meter (`µg/m³`).

This project uses the previous 24 hourly PM2.5 measurements to predict the concentration for the following hour.

## Dataset

The project uses the **Beijing Multi-Site Air-Quality Data** dataset from the UCI Machine Learning Repository.

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
- **DOI:** `10.24432/C5RK5G`
- **Time period:** March 1, 2013 – February 28, 2017
- **Frequency:** Hourly
- **Monitoring stations:** 12
- **Total records:** 420,768

The original dataset includes air pollutants and meteorological variables such as:

```text
No, year, month, day, hour, PM2.5, PM10, SO2, NO2,
CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM, station
```

This project uses only:

```text
datetime
PM2.5
```

from the **Guanyuan** monitoring station.

## Objective

The objective is to predict the next-hour PM2.5 concentration using the previous 24 hours of PM2.5 observations.

```text
Input:  24 hourly PM2.5 values
Output: Next-hour PM2.5 prediction
```

## Project Workflow

```text
Load Dataset
     ↓
Select Guanyuan Station
     ↓
Create Datetime Column
     ↓
Sort Chronologically
     ↓
Handle Missing Values
     ↓
Split Data Chronologically
     ↓
Scale PM2.5 Values
     ↓
Create 24-Hour Sequences
     ↓
Train LSTM Model
     ↓
Generate Predictions
     ↓
Inverse-Scale Predictions
     ↓
Evaluate Model Performance
     ↓
Run Streamlit Application
```

## Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset.
2. Select records from the Guanyuan station.
3. Combine the year, month, day, and hour columns into a datetime column.
4. Sort the data chronologically.
5. Convert PM2.5 values to numeric values.
6. Handle missing PM2.5 values using time-based interpolation.
7. Split the data into training and testing sets chronologically.
8. Fit `MinMaxScaler` on the training data only.
9. Transform the testing data using the same scaler.
10. Create sequences containing 24 hourly observations.

Missing values are handled using:

```python
df["PM2.5"] = df["PM2.5"].interpolate(method="time")
```

## Train/Test Split

The data is divided chronologically:

```text
80% → Training data
20% → Testing data
```

Random shuffling is avoided because the order of observations is important for time-series forecasting.

## Sequence Length

The sequence length is:

```text
24 hours
```

Since the dataset contains hourly observations, each input sequence represents the previous 24 hours and is used to predict the following hour.

## LSTM Model Architecture

```text
Input: 24 time steps × 1 feature
       ↓
LSTM(64, return_sequences=True)
       ↓
LSTM(32)
       ↓
Dense(1)
       ↓
Next-hour PM2.5 prediction
```

```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(24, 1)),
    LSTM(32),
    Dense(1)
])
```

The model is compiled for regression using the Adam optimizer and mean squared error loss:

```python
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)
```

### Training Configuration

- Maximum epochs: `30`
- Batch size: `32`
- Validation split: `10%`
- Optimizer: Adam
- Loss function: Mean Squared Error
- Training control: Early stopping

## Feature Scaling

PM2.5 values are scaled to the range `0–1` using `MinMaxScaler`.

The scaler is fitted only on the training data to prevent data leakage. It is saved as:

```text
scaler.pkl
```

The saved scaler is reused during prediction and applied to inverse-transform the model output back to the original PM2.5 scale.

## Evaluation Metrics

The model is evaluated using three regression metrics.

### Mean Absolute Error — MAE

Measures the average absolute difference between the actual and predicted values.

```text
MAE = average absolute error
```

MAE is expressed in `µg/m³`.

### Mean Squared Error — MSE

Measures the average squared difference between actual and predicted values. Larger errors have a greater effect on this metric.

```text
MSE = average squared error
```

### Root Mean Squared Error — RMSE

RMSE is the square root of MSE and is expressed in `µg/m³`.

```text
RMSE = √MSE
```

## Project Structure

```text
Proj6/
├── Dataset/
│   └── PRSA2017_Data_20130301-20170228/
├── 1-PM25Prediction.ipynb
├── 2-Prediction.ipynb
├── 3-app.py
├── model.keras
├── scaler.pkl
└── README.md
```

## Project Components

### Training Notebook

`1-PM25Prediction.ipynb`

The training notebook:

- Loads and prepares the dataset.
- Selects the Guanyuan station.
- Creates the datetime column.
- Handles missing PM2.5 values.
- Performs the chronological train/test split.
- Scales the data.
- Creates 24-hour sequences.
- Builds and trains the LSTM model.
- Evaluates model performance.
- Saves the trained model and scaler.

### Prediction Notebook

`2-Prediction.ipynb`

The prediction notebook:

- Loads `model.keras`.
- Loads `scaler.pkl`.
- Loads and preprocesses the dataset.
- Extracts the latest 24 PM2.5 measurements.
- Generates the next-hour prediction.

### Streamlit Application

`3-app.py`

The Streamlit application:

- Loads the trained LSTM model.
- Loads the saved scaler.
- Loads the Guanyuan station data.
- Displays recent PM2.5 measurements.
- Uses the latest 24 hours as model input.
- Displays the predicted next-hour PM2.5 concentration.

## Requirements

This project requires Python and the following packages:

```text
tensorflow
pandas
numpy
scikit-learn
streamlit
jupyter
ipykernel
```

Install the dependencies with:

```bash
pip install tensorflow pandas numpy scikit-learn streamlit jupyter ipykernel
```

## Usage

### Train the Model

Open the following notebook in Jupyter Notebook or Visual Studio Code:

```text
1-PM25Prediction.ipynb
```

Run the cells sequentially. The following files will be generated:

```text
model.keras
scaler.pkl
```

### Generate a Prediction

Open:

```text
2-Prediction.ipynb
```

Run the notebook after `model.keras` and `scaler.pkl` have been created.

### Run the Streamlit Application

From the project directory, run:

```bash
streamlit run 3-app.py
```

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Jupyter Notebook
- LSTM neural networks

## Dataset Citation

Chen, S. (2017). _Beijing Multi-Site Air-Quality Data_. UCI Machine Learning Repository.

DOI: `10.24432/C5RK5G`

The dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Appropriate attribution should be provided when redistributing or adapting the dataset.

## Author

Repository owner: `heart-throbb`
