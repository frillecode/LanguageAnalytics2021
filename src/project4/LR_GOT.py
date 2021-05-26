#! usr/bin/python

# Import libraries
import os
import sys
sys.path.append(os.path.join("..", ".."))
from utils import classifier_utils as clf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
import matplotlib.pyplot as plt


def crossvalidation(texts, labels, vectorizer):
    '''Function for performing cross-validation on logistic regression model
    Input:
        texts: str, text variable used to predict labels
    '''
    
    # Transform input text into tfidf-vector
    X_vect = vectorizer.fit_transform(texts)
    
    # Plot learning curves
    title = "Learning Curves (Logistic Regression)" #title of plot
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0) #create 100 different train/test-splits
    max_abs_scaler = MaxAbsScaler() #create scaling object
    X_vect_scaled = max_abs_scaler.fit_transform(X_vect) #apply scaling
    estimator = LogisticRegression(max_iter=10000, random_state=42) #perform logistic regression
    crossval_plot = clf.plot_learning_curve(estimator, title, X_vect_scaled, labels, cv=cv, n_jobs=4) #plot results
    plt.savefig(os.path.join("out", "LR_cross_validation.jpg")) #save plot
    
    return crossval_plot

def main():
    # Load data
    data = pd.read_csv(os.path.join("..", "..", "data", "project4", "Game_of_Thrones_Script.csv"))

    # Create more balanced dataset    
    data_balanced = clf.balance(pd.DataFrame({"text": data["Sentence"], "label": data["Season"]}), 1000)

    # Define texts and classification labels
    texts = data_balanced["text"].values
    labels = data_balanced["label"].values

    # Split into test/train
    X_train, X_test, y_train, y_test = train_test_split(texts,           # texts for the model
                                                        labels,          # classification labels
                                                        test_size=0.2,   # create an 80/20 split
                                                        random_state=42) # random state for reproducibility
    # Create count-vectorizer object 
    vectorizer = CountVectorizer()

    # Apply vectorizer to train and test data
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)

    # Create a pipeline which first scales the data and then performs logistic regression
    pipe = make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=10000, random_state=42))
    
    # Apply pipeline to training data
    pipe.fit(X_train_feats, y_train)

    # Use model to predict test data
    y_pred = pipe.predict(X_test_feats)
    
    # Create and save confusion matrix of results
    confusion_matrix = clf.plot_cm(y_test, y_pred, normalized=True)  
    plt.savefig(os.path.join("out", "LR_confusion_matrix.jpg"))

    # Calculate and save classifier metrics
    cm = metrics.classification_report(y_test, y_pred) 
    
    outpath = os.path.join("out", "LR_evaluation_metric.txt")
    with open(outpath, "w", encoding="utf-8") as file:
        file.write(cm)
        
    # Perform cross-validation
    crossvalidation(texts = texts, labels = labels, vectorizer=vectorizer)

    
    
# Define behaviour when called from command line
if __name__=="__main__":
    print("[INFO]: performing logistic regression classification")
    main()
    print("[INFO]: DONE!")