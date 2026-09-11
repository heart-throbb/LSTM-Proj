# Project 1: IMDB Sentiment Analysis using LSTM

This project implements a **Long Short-Term Memory (LSTM)** neural network for movie review sentiment classification.

The model classifies movie reviews into two categories:

- `0` → Negative
- `1` → Positive

## Project Overview

The project uses the **IMDB movie review dataset** provided by TensorFlow/Keras.

The reviews are converted into integer sequences, padded to a fixed length, passed through an **Embedding layer**, and then processed by an **LSTM layer**.

The final **Dense layer with sigmoid activation** produces the probability of a positive review.

### Complete Pipeline

```text
                IMDB Review
                     │
                     ▼
             Integer Encoding
                     │
                     ▼
              Padding to 200
                     │
                     ▼
                Embedding
                     │
                     ▼
                  LSTM
                     │
                     ▼
                 Dense(1)
                     │
                     ▼
                Sigmoid
                     │
                     ▼
            Probability 0–1
                     │
              ┌──────┴──────┐
              ▼             ▼
           < 0.5          >= 0.5
              │             │
              ▼             ▼
          Negative       Positive
```

## Model Architecture

```text
Input
  ↓
Embedding
  ↓
LSTM
  ↓
Dense(1)
  ↓
Sigmoid
  ↓
Positive / Negative
```

## Model Configuration

| Parameter               |                 Value |
| ----------------------- | --------------------: |
| Dataset                 |                  IMDB |
| Vocabulary Size         |                10,000 |
| Maximum Sequence Length |                   200 |
| Embedding Dimension     |                   128 |
| LSTM Units              |                    64 |
| Optimizer               |                  Adam |
| Loss Function           |   Binary Crossentropy |
| Output Activation       |               Sigmoid |
| Problem Type            | Binary Classification |

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Streamlit

## Project Structure

```text
Proj1/
│
├── 1-SentimentAnalysis.ipynb
├── 2-Prediction.ipynb
├── 3-app.py
├── model.keras
└── README.md
```

## Requirements

Install the required Python packages:

```bash
pip install tensorflow numpy streamlit
```

Or install them individually:

```bash
pip install tensorflow
pip install numpy
pip install streamlit
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project

```bash
cd Proj1
```

### 3. Install Dependencies

```bash
pip install tensorflow numpy streamlit
```

### 4. Run the Streamlit Application

```bash
streamlit run 3-app.py
```

The Streamlit application will open in your browser.

## Example Predictions

### Positive Review

**Input:**

```text
This movie was amazing and I really enjoyed it.
```

**Output:**

```text
Sentiment: Positive
```

### Negative Review

**Input:**

```text
This movie was boring and disappointing.
```

**Output:**

```text
Sentiment: Negative
```

## Concepts Learned

This project demonstrates the following concepts:

- Text preprocessing
- Tokenization
- Integer encoding
- Sequence padding
- Word embeddings
- LSTM networks
- Binary classification
- Sigmoid activation
- Binary crossentropy
- Model training
- Model evaluation
- Model saving and loading
- Streamlit deployment

## Why LSTM?

A traditional feed-forward neural network does not naturally handle sequential information.

An LSTM is designed for sequential data such as text because it can maintain information from previous words while processing a sequence.

For example:

```text
"The movie was not good"
```

The word **"not"** changes the meaning of **"good"**.

LSTM can learn relationships between words across a sequence and use this information for classification.

## SimpleRNN vs LSTM

### SimpleRNN

```text
Input
  ↓
SimpleRNN
  ↓
Dense
```

### LSTM

```text
Input
  ↓
LSTM
  ↓
Dense
```

The important difference is that **LSTM is internally more sophisticated than SimpleRNN**.

LSTM uses:

- Cell state
- Forget gate
- Input gate
- Output gate

These mechanisms help the network control which information should be:

- Forgotten
- Stored
- Updated
- Passed to the next step

This helps LSTM handle longer-term dependencies more effectively than a basic SimpleRNN.

## Interview Key Point

The complete process can be summarized as:

```text
Raw Text
   ↓
Integer Encoding
   ↓
Padding
   ↓
Embedding Layer
   ↓
LSTM Layer
   ↓
Dense Layer
   ↓
Sigmoid
   ↓
Probability
   ↓
Positive / Negative
```

For binary sentiment classification:

```text
Probability < 0.5  → Negative
Probability >= 0.5 → Positive
```

The sigmoid activation produces a value between `0` and `1`, which can be interpreted as the model's estimated probability that the review belongs to the positive class.

## Model Output

The model produces a probability:

```text
0.00 ───────────────────────────── 1.00
 │                                  │
Negative                         Positive
```

For example:

```text
0.12 → Negative
0.31 → Negative
0.52 → Positive
0.87 → Positive
0.96 → Positive
```

## Project Goal

The main goal of this project is to understand how **LSTM networks can be applied to Natural Language Processing (NLP)** tasks, particularly binary sentiment classification.

This project also provides practical experience with:

- Preparing text data for neural networks
- Building an LSTM model using TensorFlow/Keras
- Saving a trained model
- Loading a trained model for prediction
- Creating a simple Streamlit interface for deployment

## Author

Repository owner: `heart-throbb`
