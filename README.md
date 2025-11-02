# Deep Averaging Network (DAN) for Sentiment Analysis

A PyTorch implementation of a DAN model for binary sentiment classification.

## Setup

1. Install Python 3.8+
2. Run:
   ```bash
   pip install -r requirements.txt
   python -c "import stanza; stanza.download('en')"
   python sentiment_analysis.py


#About dataset
It contains sentences labelled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants


Format:
sentence \t score \n

Details:
Score is either 1 (for positive) or 0 (for negative)	

The sentences come from three different websites/fields:
imdb.com
amazon.com
yelp.com


Model Architecture
A Deep Averaging Network (DAN) works by:

1. Embedding words from each sentence.
2. Averaging the word embeddings.
3. Passing the average through fully connected layers with dropout.
4. Output class scores for sentiment prediction.

Layers:
Embedding
Linear (hidden) → ReLU
Dropout
Linear (hidden//2) → ReLU
Linear (output)


Workflow :
1. Preprocessing
Load and label the data from text files.
Tokenize each sentence using stanza.
Build a vocabulary (<PAD>, <UNK>, and all other tokens).
Convert tokens to indices and pad/truncate sequences to the same length.

2. Dataset Preparation
Split data into training and test sets using train_test_split.
Use TensorDataset and DataLoader to batch the data.

3. Training
Train the model using CrossEntropyLoss with class weights to handle imbalance.
Optimizer: Adam
Epochs: 10
Batch Size: 64

4. Evaluation
Evaluate model accuracy and print a detailed classification report using sklearn.