# Bike Sharing Demand Prediction Using LSTM

A univariate time-series forecasting project that predicts the next hour's bike-rental demand with a Long Short-Term Memory (LSTM) neural network. Given the previous 24 hourly rental counts, the model forecasts the total number of rentals for the following hour.

The repository includes the training and prediction notebooks, trained model, fitted scaler, dataset, and a Streamlit interface for interactive predictions.

## Objective

```text
Input:  rental counts from the previous 24 hours
Output: predicted total bike-rental count for the next hour
```

Only the historical total rental count (`cnt`) is used as the model feature. Weather, season, and calendar fields included in the original dataset are not used by this implementation.

## Dataset

The project uses the **Bike Sharing Dataset** collected from the Capital Bikeshare system in Washington, D.C., USA.

- UCI Machine Learning Repository: [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), DOI: `10.24432/C5W894`
- File used for modelling: `Dataset/hour.csv`
- Granularity: hourly
- Records: 17,379
- Time period: 2011–2012
- Target column: `cnt` — total rentals, including casual and registered users

The original dataset also contains calendar and weather-related fields. See [`Dataset/Readme.txt`](Dataset/Readme.txt) for its full description, citation, and licensing information.

## Approach

The training workflow is designed to preserve time order:

1. Load `hour.csv` and combine `dteday` and `hr` into a datetime index.
2. Sort the observations chronologically and select the `cnt` series.
3. Split the series chronologically: 80% training data and 20% test data.
4. Fit `MinMaxScaler` on the training values only, then transform the test values with the same scaler.
5. Build sliding windows of 24 hourly values.
6. Train a stacked LSTM network and restore the best validation weights with early stopping.
7. Inverse-transform predictions and evaluate them in bike rentals per hour.

The train/test split and test evaluation remain chronological. During fitting, Keras shuffles the training sequences by default; test observations are not used for training.

## Model Architecture

```text
Input: 24 time steps × 1 feature
       ↓
LSTM(64, return_sequences=True)
       ↓
LSTM(32)
       ↓
Dense(1)
       ↓
Next-hour rental-count prediction
```

The model is trained with the Adam optimizer and mean squared error (MSE) loss.

### Training Configuration

- Sequence length: 24 hours
- Train/test split: 80% / 20%, chronological
- Validation split: 10% of the training sequences
- Maximum epochs: 30
- Batch size: 32
- Early stopping patience: 5 epochs
- Scaling: `MinMaxScaler` with a 0–1 range

## Evaluation

The saved model was evaluated on 3,476 chronological test sequences. Results recorded in the training notebook are:

| Metric                 |           Result |
| ---------------------- | ---------------: |
| Test loss (scaled MSE) |          0.00352 |
| MAE                    | 37.73 bikes/hour |
| MSE                    |          3220.19 |
| RMSE                   | 56.75 bikes/hour |

Results can vary slightly when the model is retrained because neural-network training is stochastic.

## Project Structure

```text
Proj7/
├── Dataset/
│   ├── hour.csv                 # Hourly bike-sharing data used by the model
│   ├── day.csv                  # Daily version of the source dataset
│   └── Readme.txt               # Dataset documentation and citation
├── 1-BikeDemandPrediction.ipynb # Training and evaluation workflow
├── 2-Prediction.ipynb           # Prediction using saved artifacts
├── 3-app.py                     # Streamlit application
├── model.keras                  # Trained LSTM model
├── scaler.pkl                   # Fitted MinMaxScaler
└── README.md
```

## Requirements

Use Python 3.10 or later and install the following packages. No separate `requirements.txt` file is required for this project.

```bash
pip install tensorflow pandas numpy scikit-learn streamlit jupyter ipykernel
```

Packages used:

- `tensorflow` — LSTM model creation, training, and inference
- `pandas` — data loading and time-series preparation
- `numpy` — array operations and metric calculation support
- `scikit-learn` — `MinMaxScaler`, MAE, and MSE metrics
- `streamlit` — interactive web application
- `jupyter` and `ipykernel` — running the notebooks

`pickle` is also used to load and save the scaler; it is part of Python's standard library and does not need to be installed.

## Usage

Run all commands from the `Proj7` directory so the model, scaler, and dataset paths resolve correctly.

```bash
cd LSTMRNN/Proj7
```

### Train and evaluate

Open `1-BikeDemandPrediction.ipynb` in Jupyter Notebook or VS Code and run the cells in order. This creates or replaces:

```text
model.keras
scaler.pkl
```

### Make a notebook prediction

Open and run `2-Prediction.ipynb`. It loads the saved model and scaler, takes the most recent 24 hourly counts from `Dataset/hour.csv`, and predicts the next hour's demand.

### Launch the Streamlit app

```bash
streamlit run 3-app.py
```

The app displays basic dataset information and the latest ten rental counts. Select **Predict Next Hour** to generate a forecast from the latest 24 observations.

## Dataset Attribution

Fanaee-T, H. and Gama, J. (2013). _Event labeling combining ensemble detectors and background knowledge_. Progress in Artificial Intelligence. https://doi.org/10.1007/s13748-013-0040-3

The dataset is catalogued by the UCI Machine Learning Repository under DOI `10.24432/C5W894`. The bundled [`Dataset/Readme.txt`](Dataset/Readme.txt) requests citation of the associated paper but does not specify a reuse license. Check the repository's current terms before redistributing the dataset, and retain the attribution when adapting this project.

## Author

Repository owner: `heart-throbb`
