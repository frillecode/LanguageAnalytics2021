#! usr/bin/python

# Import libraries
import sys, os
import argparse
sys.path.append(os.path.join("..", ".."))
from utils import classifier_utils as clf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
import matplotlib.pyplot as plt


def vectorize():
    '''Creates a tf-idf vectorizer object
    '''
    # Create tfidf-vectorizer object 
    vectorizer = TfidfVectorizer(ngram_range = (1,2),    # unigrams and bigrams (1 word and 2 word units)
                                lowercase =  True,       # transform to lowercase
                                max_df = 0.95,           # remove very common words
                                min_df = 0.01,           # remove very rare words 
                                max_features = 100)      # keep only top 100 features
    return vectorizer


class PhilosophyLR:
    '''This class is desgined for performing logistic regression classification of authors of philosophical texts
    '''
    def __init__(self, args):
        self.args = args

    def prepare_data(self, data):
        '''Function for preprocessing and preparing data for LR-model
        '''
        print("[INFO]: Preparing data...")

        # Create more balanced dataset 
        df_balanced = clf.balance(pd.DataFrame({"text": data["sentence"], "label": data["author"]}), 10000)

        self.text = df_balanced["text"]
        self.labels = df_balanced["label"]

        # Split into test/train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.text,           # texts for the model
                                                                                self.labels,         # classification labels
                                                                                test_size=0.2,       # create a 80/20 split
                                                                                random_state=42)     # random state for reproducibility
        
        # Create tfidf-vectorizer object 
        self.vectorizer = vectorize()

        # Apply vectorizer to train and test data
        self.X_train_vect = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vect = self.vectorizer.transform(self.X_test)


    def grid_search(self): #This function relies code made in class and has been modified for the purpose of this project
        '''Function for performing grid-search on logistic regression model
        '''
        print("[INFO]: Performing grid-search...")
        # Initialise the default model, here given the name 'classifier'
        pipe = Pipeline([('classifier' , LogisticRegression())])

        # Set tunable parameters for grid search
        penalties = ['l1', 'l2'] # different regularization parameters
        C = [1.0, 0.1, 0.01]     # different regularization 'strengths'
        solvers = ['liblinear']  # different solvers 

        # Create parameter grid 
        parameters = dict(classifier__penalty = penalties,  
                        classifier__C = C,
                        classifier__solver = solvers)

        # Choose metric on which we want to optimise
        score = self.args['optimize_metric']

        print(f"# Tuning hyper-parameters for {score}")
        print()
        
        # Initialise gridsearch with predefined parameters
        clf = GridSearchCV(pipe, 
                        parameters, 
                        scoring= f"{score}_weighted",
                        cv=10) # use 10-fold cross-validation
        # Fit to data
        clf.fit(self.X_train_vect, self.y_train)
        

        # Save best results on training data
        self.best_params = clf.best_params_

        # Print values
        print("Grid scores on training data:")
        print()
        means = clf.cv_results_['mean_test_score'] # get all means
        stds = clf.cv_results_['std_test_score']  # get all standard deviations
        params = clf.cv_results_['params'] # get parameter combinations

        # print means, standard deviations , and parameters for all runs
        i = 0
        for mean, stdev, param in zip(means, stds, params):
            # 2*standard deviation covers 95% of the spread - check out the 68–95–99.7 rule
            print(f"Run {i}: {round(mean,3)} (SD=±{round(stdev*2, 3)}), using {param}")
            i += 1
        # print best
        print()
        print("Best parameters set found on training data:")
        print()
        print(self.best_params)
        print()


    def modelling(self):
        '''Function for fitting LR-model to data 
        '''
        if self.args['grid_search'] == 1:
            print("[INFO]: Fitting model with grid-search parameters...")
            # Create a pipeline which first scales the data and then performs logistic regression (using parameters from grid-search)
            pipe = make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=10000, 
                                                                    C = self.best_params['classifier__C'],
                                                                    penalty = self.best_params['classifier__penalty'],
                                                                    solver = self.best_params['classifier__solver'],
                                                                    random_state=42))
        else: 
            print("[INFO]: Fitting model with default parameters...")
            # Create a pipeline which first scales the data and then performs logistic regression (using default parameters)
            pipe = make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=10000, random_state=42))
            
        # Apply pipeline to training data
        pipe.fit(self.X_train_vect, self.y_train)

        # Use model to predict test data
        self.y_pred = pipe.predict(self.X_test_vect)
        
        # Create and save confusion matrix of results
        confusion_matrix = clf.plot_cm(self.y_test, self.y_pred, normalized=True)  
        plt.savefig(os.path.join("out", "confusion_matrix.jpg"))

        # Calculate and save classifier metrics
        cm = metrics.classification_report(self.y_test, self.y_pred) 
        print()
        print("Classification report:")
        print(cm)

        outpath = os.path.join("out", "evaluation_metric.txt")
        with open(outpath, "w", encoding="utf-8") as file:
            file.write(cm)
            
        # Perform cross-validation
        print("[INFO]: Performing cross-validation...")
        #transform input text into tfidf-vector
        X_vect = self.vectorizer.fit_transform(self.text)
        
        #plot learning curves
        title = "Learning Curves (Logistic Regression)" #title of plot
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0) #create 100 different train/test-splits
        max_abs_scaler = MaxAbsScaler() #create scaling object
        X_vect_scaled = max_abs_scaler.fit_transform(X_vect) #apply scaling
        estimator = LogisticRegression(max_iter=10000, random_state=42) #perform logistic regression
        crossval_plot =clf.plot_learning_curve(estimator, title, X_vect_scaled, self.labels, cv=cv, n_jobs=4) #plot results
        plt.savefig(os.path.join("out", "cross_validation.jpg")) #save plot

def main():
    ap = argparse.ArgumentParser(description="[INFO] This script perform Logistic Regression to classify authors of philosophical texts based on tf-idf vectors of the texts.")
    # Argument for specyfing whether to do grid-search or not
    ap.add_argument("-gs", 
                "--grid_search", 
                required=False, 
                type=int,
                default=1,
                help="int, whether or not to use grid-search for setting optimal parameters in model (0/1)") 
    # Argument to specify which metrics to optimize in grid-search
    ap.add_argument("-om",
                "--optimize_metric",
                required=False,
                type=str,
                choices=['precision', 'recall', 'f1'],
                default='f1',
                help="str, metrics to optimize when doing grid-search ('precision', 'recall', or 'f1')")

    args = vars(ap.parse_args())

    # Load data
    data = pd.read_csv(os.path.join("..", "..", "data", "project3", "sentences.csv")) 

    # Run analysis
    LR = PhilosophyLR(args)
    LR.prepare_data(data)
    if args['grid_search'] == 1:
        LR.grid_search()
    LR.modelling()


# Define behaviour when called from command line
if __name__=="__main__":
    print("[INFO]: Performing logistic regression classification")
    main()
    print("[INFO]: DONE!")
    