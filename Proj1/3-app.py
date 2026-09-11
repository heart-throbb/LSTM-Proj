import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LENGTH = 200


@st.cache_resource
def load_sentiment_model():
    return load_model("model.keras")


model = load_sentiment_model()
word_index = imdb.get_word_index()


def encode_review(review):
    words = review.lower().split()
    encoded_review = []
    for word in words:
        index = word_index.get(word)
        if index is not None and index < VOCAB_SIZE - 3:
            encoded_review.append(index + 3)
        else:
            encoded_review.append(2)
    return encoded_review


def predict_sentiment(review):
    encoded_review = encode_review(review)
    padded_review = pad_sequences(
        [encoded_review], maxlen=MAX_LENGTH, padding="post", truncating="post"
    )
    prediction = model.predict(padded_review, verbose=0)[0][0]
    if prediction >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, float(prediction)


st.title("IMDB Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review and the LSTM model "
    "will predict whether it is positive or negative."
)

review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was amazing and I really enjoyed it.",
)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        sentiment, probability = predict_sentiment(review)
        if sentiment == "Positive":
            st.success("Sentiment: Positive")
        else:
            st.error("Sentiment: Negative")
        st.write(f"Positive Probability: {probability:.2%}")
