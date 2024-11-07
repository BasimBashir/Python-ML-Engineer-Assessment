# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

# Loading IMDb word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Preprocessing text for prediction
def preprocess_text(text, vocab_size=10000):
    # Tokenizing input text to match model's input format
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.word_index = word_index  # Load IMDb word index into tokenizer
    tokens = tokenizer.texts_to_matrix([text], mode='binary')  # Binary bag-of-words vector
    return tokens

# Loading the trained model
def load_trained_model(filepath='sentiment_model.h5'):
    return load_model(filepath)

# Predicting sentiment
def predict_sentiment(model, text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment, prediction

# Main function
def main():
    # Loading the trained model
    model = load_trained_model()
    
    # Taking user input
    text = input("Enter a movie review: ")
    
    # Predicting sentiment
    sentiment, score = predict_sentiment(model, text)
    print(f"Review: {text}")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {score:.2f})")

if __name__ == '__main__':
    main()
