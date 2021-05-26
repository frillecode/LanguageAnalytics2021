# Project 3: (Un)supervised machine learning
This project was developed as a solution to assignment 5 set by our teacher, Ross Deans Kristensens-McLachlan, during the course. A Github repository which contains all of the code in relation to my solution to the assignment can be found here: 
https://github.com/frillecode/LanguageAnalytics2021/tree/main/src/project3

## Project description 
### Applying (un)supervised machine learning to text data
For this task, you will pick your own dataset to study.

This dataset might be something to do with COVID-19 discourse on Reddit; IMDB reviews; newspaper headlines; whatever it is that catches your eye. However, I strongly recommend using a text dataset from somewhere like Kaggle - https://www.kaggle.com/datasets

When you've chosen the data, do one of the following tasks. One of them is a supervised learning task; the other is unsupervised.

EITHER  
- Train a text classifier on your data to predict some label found in the metadata.   
OR   
- Train an LDA model on your data to extract structured information that can provide insight into your data. For example, maybe you are interested in seeing how different authors cluster together or how concepts change over time in this dataset.


## Methods
I chose to analyse a [dataset](https://www.kaggle.com/christopherlemke/philosophical-texts?select=sentences.csv) consisting of texts from different philosophers to see whether structures of the texts could be used the predict the author. The file that I chose is a collection of sentences extracted from the text files along with a column indicating the author of the text. As the number of texts for each author varied quite greatly, I created a balanced dataset consisting of 1000 texts for each author.   

I trained a multinomial logistic regression classification model to predict the author of a philosophical text based on a tfid-vector of the words in the texts (Scott, 2019). For this, [sklearn](https://scikit-learn.org/stable/index.html) was used. I scaled the data using the inbuilt MaxAbsScaler() from sklearn (dividing by max value for each feature to get values between -1 and 1), which is a solution to sparse data suggested in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/preprocessing.html). For the tuning of hyperparameters, I included the option of performing grid-search to see how different parameters affect model performance across metrics. By default, the script performs grid search to optimize the weighted F1-score and uses the resulting best parameters in the model. It is possible to choose between three metrics to optimize ('recall', 'precision', 'f1') through the command-line. In addition, it is possible to parse an argument to turn grid-search off, in which case the model uses the default parameters of sklearn's LogisticRegression(). 
After fitting the model, the script performs 10-fold cross-validation to validate results using different train/test-splits.   

The scripts saves a confusion matrix, classifier metrics and cross-validation results in the 'out/'-folder.


## Usage
The structure of the files belonging to this project is as follows:
```bash
LanguageAnalytics2021/  
├── data/ #data
│   └── project3/
│   │   └── sentences.csv 
├── src/ #scripts
│   └── project3/
│   │   └── out/  #results
│   │   └── LR_philosophicalTexts.py  
├── utils/  #utility functions 
│   └── *.py  
```
### Data
Data file too large to push to GitHub. Downloaded from [here](https://www.kaggle.com/christopherlemke/philosophical-texts?select=sentences.csv) and upload to the data-folder belonging to this project ('../../data/project3/').

### Cloning repo and installing dependencies 
To run the script, I recommend cloning this repository and installing relevant dependencies in a virtual environment:  

```bash
$ git clone https://github.com/frillecode/LanguageAnalytics2021
$ cd LanguageAnalytics2021
$ bash ./create_venv.sh #use create_venv_win.sh for windows
```

If you run into issues with some libraries/modules not being installed correctly when creating the virtual environment, install these manually by running the following:  
```bash
$ cd LanguageAnalytics2021
$ source cds-lang/bin/activate
$ pip install {module_name}
$ deactivate
```

### Running scripts
After updating the repo (see above), you can run the .py-files from the command-line by writing the following:
``` bash
$ cd LanguageAnalytics2021
$ source cds-lang/bin/activate
$ cd src/project3
$ python3 LR_philosophicalTexts.py
```

The script takes different optional arguments that can be specified in the command-line:
```bash
- "-gs", "--grid_search", required=False, type=int, choices=[0,1], default=1, help="int, whether or not to use grid-search for setting optimal parameters in model (0 or 1)") 
- "-om", "--optimize_metric", required=False, type=str, choices=['precision', 'recall', 'f1'], default='f1', help="str, metrics to optimize when doing grid-search ('precision', 'recall', or 'f1')")
```

By default, the script performs grid-search to optimize the weighted f1-score of the model. It is possible to choose to optimize for either 'recall' or 'precision' instead. To do so, run e.g.: 
``` bash
$ python3 LR_philosophicalTexts.py -om 'precision'
```
It is also possible to turn off grid-search and use the default parameters for the logistic regression by running:
``` bash
$ python3 LR_philosophicalTexts.py -gs 0
```

You can get more information on the optional arguments that can be parsed by running:
``` bash
$ python3 LR_philosophicalTexts.py --help
```

## Discussion of results
The results currently seen in the 'out'-folder are from performing the logistic regression using the parameters of a grid-search optimizing for weighted f1-score. I played around with some different choices for analysis (e.g. the thresholds for removing common/uncommon words) to see how this would affect the performance of the model. Setting a lower threshold for removing rare words (those that appear in less than 0.1% of the texts) seemed to increase the accuracy of the model. However, the most succesful model had an accuracy of 0.57 indicating that it was not successfulin predicting author from a tfid-vector of words in the texts. The F1-scores ranged between 0.43 to 0.62 for the different authors, implying that the model differed in how successful it was for predicting different authors. Inspecting the learning curves from cross-validation, we see that both the training and cross-validation scores are quite low. The scores of both curves start within the range of 0.55-0.60 with the training score decreasing and the cross-validation score increasing with more training time. The gap between the curves becomes smaller, however, it seems that the curves have not quite converged, indicating that the model might benefit from more data. 



## References

Scott, W., (2019), "TF-IDF from scratch in python on real world dataset.", towards data science, https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089