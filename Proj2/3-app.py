import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


@st.cache_resource
def load_lstm_model():
    return load_model("model.keras")


model = load_lstm_model()


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as file:
        return pickle.load(file)


tokenizer = load_tokenizer()
max_sequence_length = model.input_shape[1] + 1


def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_length - 1, padding="pre"
    )
    predicted_probabilities = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted_probabilities, axis=-1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


def generate_text(seed_text, next_words):
    result = seed_text
    for _ in range(next_words):
        next_word = predict_next_word(result)
        if next_word is None:
            break
        result += " " + next_word
    return result


st.title("LSTM Next Word Prediction")
st.write("Enter a starting phrase and the LSTM model " "will predict the next words.")

seed_text = st.text_input(
    "Enter starting text:", placeholder="Example: machine learning"
)
number_of_words = st.slider(
    "Number of words to generate:", min_value=1, max_value=10, value=5
)

if st.button("Generate Text"):
    if seed_text.strip() == "":
        st.warning("Please enter some starting text.")
    else:
        generated_text = generate_text(seed_text, number_of_words)
        st.subheader("Generated Text")
        st.write(generated_text)
