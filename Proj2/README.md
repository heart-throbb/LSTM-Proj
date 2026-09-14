# LSTM Next Word Prediction

A simple next-word prediction project built with TensorFlow/Keras and Streamlit. The model is trained on a small text corpus related to machine learning, artificial intelligence, deep learning, Python, and data science, then used to generate the next word or continue a short input phrase.

This project demonstrates the complete workflow for a basic language modeling task:

- Preparing text data for sequence modeling
- Tokenizing sentences with Keras `Tokenizer`
- Creating n-gram input sequences
- Padding sequences for fixed-length model input
- Training an LSTM-based neural network
- Saving and loading a trained model
- Building an interactive Streamlit app for text generation

## Demo

The Streamlit app allows users to enter a starting phrase and choose how many words to generate.

Example inputs:

```text
machine learning
python is
artificial intelligence
```

Example output:

```text
artificial intelligence is changing the world
```

Output may vary depending on the trained model and the input phrase.

## Project Structure

```text
Proj2/
├── 1-NextWordPrediction.ipynb  # Model training notebook
├── 2-Prediction.ipynb          # Model loading and prediction notebook
├── 3-app.py                    # Streamlit web app
├── model.keras                 # Trained Keras model
├── tokenizer.pkl               # Saved tokenizer
└── README.md                   # Project documentation
```

## Model Overview

The model uses a simple neural network architecture for next-word prediction:

- `Embedding` layer to convert word indexes into dense vector representations
- `LSTM` layer to learn sequence patterns
- `Dense` output layer with `softmax` activation to predict the next word

The training data is converted into n-gram sequences. For example, a sentence such as:

```text
machine learning is very useful
```

is transformed into multiple training examples so the model can learn to predict the next token from previous tokens.

## Technologies Used

- Python
- NumPy
- TensorFlow / Keras
- Streamlit
- Pickle
- Jupyter Notebook

## Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd <your-repository-name>/14-LSTMRNN/Proj2
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install numpy tensorflow streamlit notebook
```

## How to Run the App

From the `Proj2` directory, run:

```bash
streamlit run 3-app.py
```

Streamlit will start a local development server and show a URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser, enter a starting phrase, select the number of words to generate, and click **Generate Text**.

## How to Train the Model

Open and run the training notebook:

```bash
jupyter notebook 1-NextWordPrediction.ipynb
```

The notebook performs the following steps:

1. Defines a small training corpus.
2. Fits a Keras tokenizer on the corpus.
3. Creates n-gram input sequences.
4. Pads all sequences to the same length.
5. Splits each sequence into input features and target labels.
6. Trains an LSTM model.
7. Saves the trained model as `model.keras`.
8. Saves the tokenizer as `tokenizer.pkl`.

After training, the saved model and tokenizer can be used by both the prediction notebook and the Streamlit app.

## How Prediction Works

The prediction pipeline follows these steps:

1. Convert the user input text into token indexes using the saved tokenizer.
2. Pad the token sequence to match the model input length.
3. Use the trained model to predict probabilities for the next word.
4. Select the word with the highest predicted probability.
5. Repeat the process to generate multiple words.

## Limitations

This is an educational project trained on a small custom corpus. Because of that:

- The model performs best on words and phrases similar to the training data.
- It may repeat words when generating longer text.
- It is not intended to behave like a large language model.
- Predictions are limited by the vocabulary learned during training.

For better results, the model can be trained on a larger and more diverse text dataset.

## Future Improvements

Possible improvements include:

- Training on a larger text corpus
- Adding validation data and evaluation metrics
- Using top-k sampling instead of always selecting the highest-probability word
- Adding temperature-based text generation
- Improving the Streamlit UI with prediction confidence scores
- Packaging dependencies in a `requirements.txt` file

## License

This project is intended for learning and portfolio use. Add a license file before publishing if you want to define specific usage permissions.

## Author

Repository owner: `heart-throbb`
