# Traffic Volume Prediction using LSTM

This project builds a Long Short-Term Memory (LSTM) neural network to predict the next hour of traffic volume from the previous 24 hourly traffic-volume readings.

The project uses the **Metro Interstate Traffic Volume** dataset from the UCI Machine Learning Repository. The implementation focuses on univariate time-series forecasting, using only the historical `traffic_volume` column as input.

## Project Overview

Traffic volume changes over time and often follows daily patterns. This project uses those sequential patterns to train an LSTM model that forecasts the next hour's traffic volume.

The workflow includes:

- Loading and preparing the dataset
- Sorting records chronologically
- Selecting the `traffic_volume` time series
- Splitting the data into training and testing sets
- Scaling values with `MinMaxScaler`
- Creating 24-hour input sequences
- Training a stacked LSTM model
- Evaluating predictions with regression metrics
- Saving the trained model and scaler
- Running predictions through a notebook or Streamlit app

## Dataset

**Dataset:** Metro Interstate Traffic Volume  
**Source:** UCI Machine Learning Repository  
**DOI:** `10.24432/C5X60B`

The dataset contains 48,204 hourly records collected from an Interstate 94 monitoring station between Minneapolis and St. Paul, Minnesota.

Original dataset columns:

```text
holiday
temp
rain_1h
snow_1h
clouds_all
weather_main
weather_description
date_time
traffic_volume
```

This implementation uses:

```text
date_time
traffic_volume
```

`traffic_volume` is the target variable and the only feature used by the LSTM model.

## Problem Statement

Given the previous 24 hours of traffic-volume observations, predict the traffic volume for the next hour.

```text
Input:  24 previous hourly traffic-volume values
Output: next-hour traffic-volume prediction
```

## Project Structure

```text
Proj5/
|-- Dataset/
|   `-- Metro_Interstate_Traffic_Volume.csv
|-- 1_TrafficPrediction.ipynb
|-- 2-Prediction.ipynb
|-- 3-app.py
|-- model.keras
|-- scaler.pkl
`-- README.md
```

## Methodology

The dataset is first sorted by `date_time` to preserve the natural order of the time series. The project then extracts the `traffic_volume` column, reshapes it for modeling, and splits it chronologically into training and testing data.

The data is not randomly shuffled because this is a forecasting task. The model must learn from past observations and be evaluated on later observations.

```text
80% training data
20% testing data
```

`MinMaxScaler` is fitted on the training data only, then reused for the test data and future predictions. The fitted scaler is saved as `scaler.pkl`.

## Sequence Creation

The model uses a sequence length of 24:

```text
24 hourly observations => predict the next hour
```

Each training example has the shape:

```text
(24, 1)
```

where `24` is the number of time steps and `1` is the traffic-volume feature.

## Model Architecture

The project uses a stacked LSTM model:

```python
model = Sequential(
    [
        LSTM(64, return_sequences=True, input_shape=(24, 1)),
        LSTM(32),
        Dense(1),
    ]
)
```

Model summary:

```text
LSTM(64)  => returns sequences
LSTM(32)  => final sequence representation
Dense(1)  => predicted traffic volume
```

The model is compiled with:

```python
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)
```

Training uses:

- Adam optimizer
- Mean Squared Error loss
- Batch size of 32
- Maximum of 30 epochs
- 10% validation split
- Early stopping with best-weight restoration

The trained model is saved as:

```text
model.keras
```

## Evaluation

The trained model is evaluated on the test set using regression metrics.

Results from the training notebook:

```text
Test Loss: 0.0029022559
MAE:       265.47 vehicles/hour
MSE:       153814.8944
RMSE:      392.19 vehicles/hour
```

## Notebooks

### 1_TrafficPrediction.ipynb

This notebook contains the full training pipeline:

- Load the dataset
- Inspect columns and missing values
- Convert `date_time` to datetime
- Sort the data chronologically
- Select `traffic_volume`
- Create train/test splits
- Scale the data
- Generate 24-hour sequences
- Build and train the LSTM model
- Evaluate the model
- Save `model.keras`
- Save `scaler.pkl`

### 2-Prediction.ipynb

This notebook loads the saved model and scaler, takes the latest 24 traffic-volume readings from the dataset, and predicts the next hour of traffic volume.

Example output:

```text
Predicted next-hour traffic volume: 650 vehicles/hour
```

## Streamlit App

The project includes a small Streamlit application in `3-app.py`.

The app:

- Loads `model.keras`
- Loads `scaler.pkl`
- Loads the traffic dataset
- Displays dataset information
- Shows the latest traffic measurements
- Predicts the next-hour traffic volume from the latest 24 observations

Run the app with:

```bash
streamlit run 3-app.py
```

## Requirements

Install the required Python packages with:

```bash
pip install tensorflow pandas numpy scikit-learn streamlit
```

Required libraries:

```text
tensorflow
pandas
numpy
scikit-learn
streamlit
pickle
```

`pickle` is part of the Python standard library, so it does not need to be installed separately.

## How to Run

1. Open the project folder:

```bash
cd Proj5
```

2. Install the required dependencies:

```bash
pip install tensorflow pandas numpy scikit-learn streamlit
```

3. Run the training notebook if you want to retrain the model:

```text
1_TrafficPrediction.ipynb
```

4. Run the prediction notebook for a notebook-based prediction:

```text
2-Prediction.ipynb
```

5. Run the Streamlit app:

```bash
streamlit run 3-app.py
```

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Jupyter Notebook
- LSTM neural networks

## Dataset Citation

Hogue, J. (2019). **Metro Interstate Traffic Volume**. UCI Machine Learning Repository. DOI: `10.24432/C5X60B`

The dataset is available under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## Author

Repository owner: `heart-throbb`
