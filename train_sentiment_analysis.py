# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Loading and preprocessing IMDb dataset
def load_and_preprocess_data(vocab_size=10000, max_length=200):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
    
    # Convert each sequence to a binary bag-of-words vector directly
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_sequences(X_train)
    X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
    X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
    
    return X_train, y_train, X_test, y_test

# Defining the new model architecture
def build_model(input_shape=(10000,)):
    model = models.Sequential()
    model.add(layers.Dense(16, kernel_regularizer=regularizers.l1(0.001), activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, kernel_regularizer=regularizers.l1(0.001), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training the model with early stopping
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=512):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    return history

# Plotting accuracy and loss graphs
def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

# Saving the trained model
def save_model(model, filepath='sentiment_model.h5'):
    model.save(filepath)

# Main function
def main():
    # Loading data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # Splitting validation set from training data
    X_val, y_val = X_train[:5000], y_train[:5000]
    X_train, y_train = X_train[5000:], y_train[5000:]
    
    # Building and training model
    model = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plotting metrics
    plot_metrics(history)
    
    # Saving model
    save_model(model)

if __name__ == '__main__':
    main()
