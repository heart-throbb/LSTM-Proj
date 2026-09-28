# LSTM & RNN Projects

> A hands-on collection of six deep-learning projects that apply **Long Short-Term Memory (LSTM)** networks to natural-language processing and real-world time-series forecasting.

This repository documents the complete machine-learning workflow: data preparation, sequence construction, model training, evaluation, saved-model inference, and interactive deployment with Streamlit. Each project is self-contained, with training and prediction notebooks, a saved Keras model, and where needed the preprocessing artifacts required for consistent inference.

## What’s Inside

| # | Project | Task | Model Input | Output |
| --- | --- | --- | --- | --- |
| 1 | [IMDB Sentiment Analysis](Proj1/) | Binary text classification | Movie-review tokens | Positive or negative sentiment |
| 2 | [Next-Word Prediction](Proj2/) | Language modeling | Seed text | Generated continuation |
| 3 | [AAPL Stock Price Prediction](Proj3/) | Time-series regression | Previous 60 closing prices | Next closing-price estimate |
| 4 | [Household Energy Prediction](Proj4/) | Time-series regression | Previous 60 half-hour readings | Next 30-minute energy-use estimate |
| 5 | [Traffic Volume Prediction](Proj5/) | Time-series regression | Previous 24 hourly readings | Next-hour traffic-volume estimate |
| 6 | [PM2.5 Air Quality Prediction](Proj6/) | Time-series regression | Previous 24 hourly readings | Next-hour PM2.5 estimate |

## Repository Structure

```text
LSTM-Proj/
├── Proj1/  # IMDB sentiment analysis
├── Proj2/  # Next-word prediction
├── Proj3/  # AAPL stock-price forecasting
├── Proj4/  # Household energy forecasting
├── Proj5/  # Traffic-volume forecasting
├── Proj6/  # PM2.5 air-quality forecasting
└── README.md
```

Each project follows a consistent layout:

```text
ProjX/
├── 1-*.ipynb       # Training and evaluation workflow
├── 2-Prediction.ipynb
├── 3-app.py        # Streamlit application
├── model.keras     # Saved trained LSTM model
├── scaler.pkl / tokenizer.pkl  # Preprocessing artifact, when applicable
├── Dataset/         # Project data, when included
└── README.md        # Project-specific documentation
```

## Project Highlights

### 1. IMDB Sentiment Analysis

Classifies movie reviews as positive or negative using an embedding layer followed by an LSTM. Reviews are integer-encoded, padded to 200 tokens, and passed to a sigmoid output layer.

- Dataset: TensorFlow/Keras IMDB dataset
- Task: Binary classification
- Interface: Enter a review and receive a sentiment label plus positive-sentiment probability

### 2. Next-Word Prediction

Learns to predict and generate the next word from a seed phrase using n-gram sequences, an embedding layer, LSTM, and a softmax output layer.

- Task: Multi-class next-token prediction
- Interface: Generate 1–10 words from a starting phrase
- Note: This is an educational language model trained on a small custom corpus

### 3. AAPL Stock Price Prediction

Forecasts the next AAPL closing price from the preceding 60 trading days using a stacked LSTM model and MinMax scaling.

- Dataset: Historical AAPL prices (2006–2018)
- Task: One-step-ahead regression
- Evaluation: MAE, MSE, and RMSE
- Note: For educational use only—not financial or investment advice

### 4. Household Energy Consumption Prediction

Predicts the next 30-minute `Global_active_power` reading from the prior 60 observations (30 hours of history).

- Dataset: UCI Individual Household Electric Power Consumption dataset
- Task: One-step-ahead regression
- Preprocessing: Missing-value interpolation and 30-minute resampling
- Data requirement: Download the source data and place it in `Proj4/Dataset/` before running the app

### 5. Traffic Volume Prediction

Predicts the next hour of traffic volume using the previous 24 hourly observations.

- Dataset: UCI Metro Interstate Traffic Volume dataset
- Task: One-step-ahead regression
- Evaluation from the training notebook: MAE 265.47 vehicles/hour; RMSE 392.19 vehicles/hour

### 6. PM2.5 Air Quality Prediction

Predicts next-hour PM2.5 concentration at Beijing’s Guanyuan monitoring station from the previous 24 hourly readings.

- Dataset: UCI Beijing Multi-Site Air-Quality Data dataset
- Task: One-step-ahead regression
- Preprocessing: Chronological ordering, missing-value interpolation, and MinMax scaling
- Data requirement: Download the Guanyuan station file and place it in `Proj6/Dataset/` before running the app

## Core Concepts Demonstrated

- Sequence modeling with LSTM networks
- Tokenization, padding, and word embeddings
- N-gram sequence generation for language modeling
- Chronological train/test splitting for time-series data
- Feature scaling with `MinMaxScaler`
- Multi-step sequence creation for supervised learning
- Classification and regression evaluation
- Saving/loading Keras models and preprocessing artifacts
- Interactive model inference with Streamlit

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Getting Started

Clone the repository and move to this folder:

```bash
git clone <repository-url>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the shared dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn streamlit notebook matplotlib
```

## Run a Project

Every project can be explored in three ways:

1. Open the `1-*.ipynb` notebook to inspect data preparation, model training, and evaluation.
2. Open `2-Prediction.ipynb` to run saved-model inference.
3. Launch the interactive Streamlit app from the selected project directory.

For example, to run the traffic forecasting app:

```bash
cd Proj5
streamlit run 3-app.py
```

Streamlit will provide a local URL, typically `http://localhost:8501`.

## Important Notes

- Saved models expect the accompanying `scaler.pkl` or `tokenizer.pkl` files where present. Keep these files together.
- Projects 4 and 6 require source datasets that are not included because of their size. Follow each project’s README for the expected dataset path and filename.
- Forecasts use historical patterns only. They are learning demonstrations and should not be used for operational, medical, environmental, financial, or safety-critical decisions.

## Learning Path

For the smoothest progression through the repository, start with Projects 1 and 2 to learn LSTM text workflows, then move to Projects 3–6 for time-series forecasting with chronological splits, scaling, and regression metrics.


## Author

Repository owner: `heart-throbb`

---
If this repository helps your learning journey, consider giving it a star.