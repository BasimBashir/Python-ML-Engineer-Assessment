# Sentiment Analysis with IMDb Movie Reviews

This repository contains Python scripts for a machine learning project that trains a deep learning model to perform sentiment analysis on IMDb movie reviews. The project includes two scripts:
1. `train_sentiment_analysis.py` - Trains the sentiment analysis model on IMDb movie reviews and saves the trained model.
2. `inference_sentiment_analysis.py` - Uses the saved model to predict the sentiment (positive or negative) of new movie review text inputs.

## Dataset

The project uses the IMDb movie reviews dataset, an open-source dataset containing 50,000 labeled movie reviews (positive or negative sentiment). The dataset can be downloaded directly from the Keras library, so no external download steps are necessary.

## Requirements

Ensure you have Python 3.x installed, along with the following packages:
- `tensorflow`
- `numpy`
- `matplotlib`

Install dependencies with the following command:
```bash
pip install tensorflow numpy matplotlib
```

## Project Structure

- `train_sentiment_analysis.py` - Script for training the model.
- `inference_sentiment_analysis.py` - Script for making predictions on new text input.
- `README.md` - Overview and instructions.
- `requirements.txt` - List of dependencies.

## Model Architecture

The model is a deep neural network with the following layers:
1. **Dense layer** with 16 units, L1 regularization, and ReLU activation.
2. **Dropout layer** with a 50% rate to prevent overfitting.
3. **Dense layer** with 16 units, L1 regularization, and ReLU activation.
4. **Dropout layer** with a 50% rate.
5. **Output layer** with a single unit and sigmoid activation for binary sentiment classification (positive/negative).

The model is compiled with:
- **Loss function**: Binary Crossentropy
- **Optimizer**: RMSprop
- **Metrics**: Accuracy

## Training Script: `train_sentiment_analysis.py`

### Script Steps
1. **Load and preprocess data**: Loads the IMDb dataset, tokenizes the text reviews, and converts them to a binary bag-of-words matrix with a vocabulary size of 10,000.
2. **Model building**: Defines a deep neural network with dropout and regularization.
3. **Train the model**: Uses early stopping to prevent overfitting, with training stopping after 3 consecutive epochs with no validation improvement.
4. **Plot training metrics**: Accuracy and loss curves are plotted to visualize training progress and generalization.
5. **Save model**: Saves the trained model as sentiment_model.h5.

### Run the Training Script
To train the model, execute the following:
```bash
python train_sentiment_analysis.py
```
This script will train the model and display accuracy/loss curves. Once training is complete, the model is saved to `sentiment_model.h5`.

## Inference Script: `inference_sentiment_analysis.py`
### Script Steps
1. **Load trained model**: Loads the pre-trained model from `sentiment_model.h5`.
2. **Preprocess input text**: Takes a new movie review, tokenizes and pads it to match the input format used during training.
3. **Predict sentiment**: Uses the model to predict the sentiment (positive/negative) and displays the input text with the predicted sentiment.

### Run the Inference Script
To run the inference script, execute the following:
```bash
python inference_sentiment_analysis.py
```
The script will prompt for a text input (movie review), preprocess it, and output the predicted sentiment.

## Challenges and Solutions
### Challenges
1. **Overfitting**: During initial training, the model overfit quickly due to limited data processing steps and model complexity.

2. **Data preprocessing**: Handling variable-length text inputs for consistent model input shape.

### Solutions
1. **Dropout and Regularization**: Added L1 regularization and dropout layers to mitigate overfitting.

2. **Binary Bag-of-Words Encoding**: Used a binary bag-of-words representation to reduce input dimensionality, keeping essential information and simplifying preprocessing.


## Results and Evaluation
The model was able to achieve reasonable accuracy on both training and validation datasets, as visualized in the training metrics plots. The early stopping mechanism helped to prevent overfitting and improve generalization.

## Additional Notes
- **Vocabulary Size**: Limited to the top 10,000 words, which balances performance and memory efficiency.
- **Maximum Sequence Length**: Set to 200, which captures enough context for most reviews without excessive computational cost.

## Future Improvements
1. **Experiment with different architectures**: More complex models, like RNNs or LSTM-based networks, may improve accuracy for nuanced sentiment detection.
2. **Use pretrained embeddings**: Leveraging pretrained embeddings like GloVe or Word2Vec can enhance the model’s semantic understanding of text.

## Conclusion
This project demonstrates building a sentiment analysis model with TensorFlow and Keras. By implementing dropout and L1 regularization, the model achieves good performance while maintaining simplicity.

## Instructions for Running the Project
1. Clone this repository.
2. Install the required dependencies.
3. Run train_sentiment_analysis.py to train the model and save it.
4. Run inference_sentiment_analysis.py to test the model with new input text.

Happy coding!