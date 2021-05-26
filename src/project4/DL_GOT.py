#! usr/bin/python

# system tools
import os
import sys
sys.path.append(os.path.join("..", ".."))
import argparse

# data munging tools
import pandas as pd
import numpy as np
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
#from sklearn import metrics
from sklearn.metrics import classification_report

# Tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns


class CNNgot:
    '''This is a class for performing Deep Learning using a Convolutional Neural Network on text data using pre-trained word-embeddings from GloVe.
    '''
    def __init__(self, args):
        self.args = args

    def preprocessing(self, text, labels):
        '''Preprocessing function to prepare data for model
        Input:
            text: numpy.ndarray, text data
            labels: numpy.ndarray, labels for text data
        '''
        print("[INFO] Preprocessing data... ")

        self.texts = text
        self.labels = labels
        self.labelNames = set(labels)

        # 1) Split into train-test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.texts,      # texts for the model
                                                                                self.labels,     # classification labels
                                                                                test_size=0.2,   # create an 80/20 split
                                                                                random_state=42) # random state for reproducibility
        # 2) Binarize labels
        lb = LabelBinarizer()
        self.y_train = lb.fit_transform(self.y_train)
        self.y_test = lb.fit_transform(self.y_test)

        # 3) Tokenize text 
        tokenizer = Tokenizer(num_words=None)

        # fit to training data
        tokenizer.fit_on_texts(self.X_train)

        # tokenized training and test data
        self.X_train_toks = tokenizer.texts_to_sequences(self.X_train)
        self.X_test_toks = tokenizer.texts_to_sequences(self.X_test)

        # overall vocabulary size
        self.vocab_size = len(tokenizer.word_index) + 1  # (adding 1 because of reserved 0 index)

        # 4) Apply padding to ensure equal length
        self.maxlen = max([len(x) for x in self.X_train]) #max length of sentences

        # pad training data to maxlen
        self.X_train_pad = pad_sequences(self.X_train_toks, 
                                    padding='post', #adding 0s to the end of the sequence
                                    maxlen=self.maxlen)
        # pad testing data to maxlen
        self.X_test_pad = pad_sequences(self.X_test_toks, 
                                padding='post', #adding 0s to the end of the sequence
                                maxlen=self.maxlen)
        
        # 5) Create embedding matrix
        # create matrix using glove embeddings
        self.embedding_matrix = clf.create_embedding_matrix(os.path.join("..", "..", "data", "project4", "glove", f"glove.6B.{self.args['embedding_dim']}d.txt"),
                                                            tokenizer.word_index, 
                                                            self.args['embedding_dim'])      


    def create_model(self):
        '''Set up architecture for CNN model using pretrained GloVe embeddings
        '''
        print("[INFO] Building model... ")

        # Define L2 regularizer
        l2 = L2(self.args["lambda"])

        # Initialize Sequential model
        self.model = Sequential()

        # Add Embedding layer
        self.model.add(Embedding(input_dim=self.vocab_size,             # vocab size from Tokenizer()
                                output_dim=self.args['embedding_dim'],  # user defined embedding size
                                input_length=self.maxlen,               # maxlen of padded docs 
                                weights=[self.embedding_matrix],        # pretrained GloVe weights
                                trainable=False))                       # embeddings are static - not trainable
        
        # Add ConV layer (ReLU activation)
        self.model.add(Conv1D(128, 5,
                            activation='relu',
                            kernel_regularizer = l2))

        # Add MaxPool 
        self.model.add(GlobalMaxPool1D())

        # Add Dense layer (ReLU activation)
        self.model.add(Dense(48, 
                            activation='relu',
                            kernel_regularizer = l2))

        # Add droput layer
        self.model.add(Dropout(0.2))

        # Add prediction node (softmax activation)
        self.model.add(Dense(8, 
                            activation='softmax'))

        # Compile model
        self.model.compile(loss='categorical_crossentropy',
                        optimizer="adam",
                        metrics=['accuracy'])

        # Print summary
        self.model.summary()

    def fit_model(self):
        '''This function fits the model to the data and returns a classification report based on the predictions
        '''
        print("[INFO] Training model... ")

        self.H = self.model.fit(self.X_train_pad, self.y_train,
                            epochs=self.args['epochs'],
                            verbose=False,
                            validation_data=(self.X_test_pad, self.y_test),
                            batch_size=self.args['batch_size'])
        
        # Get predictions
        self.predictions = self.model.predict(self.X_test_pad, batch_size=self.args['batch_size']) 
            
        # Comparing predictions to our test labels
        class_rep = classification_report(self.y_test.argmax(axis=1),
                                        self.predictions.argmax(axis=1),
                                        target_names=self.labelNames)

        # Save classification report   
        outpath = os.path.join("out", "DL_evaluation_metric.txt")
        with open(outpath, "w", encoding="utf-8") as file:
            file.write(class_rep)

        return class_rep


    def plot_history(self): 
        ''' Plot of the model as it learns 
        '''
        # Visualize performance
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, self.args['epochs']), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.args['epochs']), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.args['epochs']), self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.args['epochs']), self.H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()
         
        # Save plot
        plot_path = os.path.join("out", "DL_performance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description="[INFO] This script is designed to predict Game of Thrones season based on the lines spoken. The script takes text data, applies a CNN-model to predict labels, saves learning curves, and saves and prints the classification report to the terminal. ")
    # Argument for specifying number of epochs
    ap.add_argument("-e", 
                "--epochs", 
                required=False, 
                type=int, 
                default=20, 
                help="int, number of epochs") 
    # Argument for specifying embedding size
    ap.add_argument("-ed", 
                "--embedding_dim", 
                required=False, 
                type=int,
                choices=[50,100,200,300], 
                default=50, 
                help="int, embedding size of pretrained GloVe embeddings (must be either: 50, 100, 200, or 300)") 
    # Argument for specifying batch size for training model
    ap.add_argument("-bs",
                "--batch_size",
                required=False,
                type=int,
                default=32,
                help="int, batch size")
    # Argument for specifying lambda value for L2 regularization
    ap.add_argument("-l",
                "--lambda",
                required=False,
                type=float,
                default=0.001,
                help="float, lambda value for L2 regularization")              

    args = vars(ap.parse_args())

    # Load data
    data = pd.read_csv(os.path.join("..", "..", "data", "project4", "Game_of_Thrones_Script.csv"))

    # Create balanced text-set based on labels
    data_balanced = clf.balance(pd.DataFrame({"text": data["Sentence"], "label": data["Season"]}), 1000)

    # Run analysis
    convNN = CNNgot(args)
    convNN.preprocessing(text = data_balanced["text"].values, labels = data_balanced["label"].values) 
    convNN.create_model() 
    cm = convNN.fit_model() 
    print(cm)
    convNN.plot_history()

# Define behaviour when called from command line
if __name__=="__main__":
    print("[INFO]: performing CNN classification")
    main()
    print("[INFO]: DONE! You can find the results in the 'out/'-folder")