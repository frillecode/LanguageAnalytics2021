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

You should formulate a short research statement explaining why you have chosen this dataset and what you hope to investigate. This only needs to be a paragraph or two long and should be included as a README file along with the code. E.g.: I chose this dataset because I am interested in... I wanted to see if it was possible to predict X for this corpus.

In this case, your peer reviewer will not just be looking to the quality of your code. Instead, they'll also consider the whole project including choice of data, methods, and output. Think about how you want your output to look. Should there be visualizations? CSVs?

You should also include a couple of paragraphs in the README on the results, so that a reader can make sense of it all. E.g.: I wanted to study if it was possible to predict X. The most successful model I trained had a weighted accuracy of 0.6, implying that it is not possible to predict X from the text content alone. And so on.

__Tips__
- Think carefully about the kind of preprocessing steps your text data may require - and document these decisions!
- Your choice of data will (or should) dictate the task you choose - that is to say, some data are clearly more suited to supervised than unsupervised learning and vice versa. Make sure you use an appropriate method for the data and for the question you want to answer
- Your peer reviewer needs to see how you came to your results - they don't strictly speaking need lots of fancy command line arguments set up using argparse(). You should still try to have well-structured code, of course, but you can focus less on having a fully-featured command line tool


__Bonus challenges__
Do both tasks - either with the same or different datasets



## Methods
 

## Usage
The structure of the repository belonging to this assignment is as follows:  
  - Data: _../../data/project3/_ 
  - Code: _LR_philosophicalTexts.py_
  - Results: _out/_ 

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





????????

You can get more information on which optional arguments can be parsed by running:
``` bash
$ python3 edge_detection.py --help
```

## Discussion of results
The resulting output-files from running the script can be found in 'out/'. 





I chose to analyse a [dataset](https://www.kaggle.com/christopherlemke/philosophical-texts?select=sentences.csv) consisting of text from different philosophers to see whether structures of the texts could be used the predict the author. The file that I chose is a collection of sentences extracted from the text files along with a column indicating the author of the text.  

I trained a logistic regression classification model to predict the author of a philosophical text based on a tfid-vector of the words in the texts. Because the data was sparse, I scaled the data using the inbuilt MaxAbsScaler() from sklearn (dividing by max value for each feature to get values between -1 and 1), which is a solution suggested in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/preprocessing.html). I played around with some different choices for analysis (e.g. the thresholds for removing common/uncommon words) to see how this would affect the performance of the model. Setting a lower threshold for removing rare words (those that appear in less than 0.1% of the texts) seemed to increase the accuracy of the model. However, the most succesful model had an accuracy of 0.6 - just above chance - indicating that it was not possible to predict author from a tfid-vector of words in the texts. The f1-scores ranged between 0.39 to 0.74 for the different authors, implying that the model differed quite a lot in how successful it was for predicting different authors. You can find a confusion matrix and classifier metrics for this model in the _out_-folder. To validate the results, I performed cross-validation to get average performance using different test-train splits. The results of this can also be found in the _out_-folder. Inspecting the learning curves, it seems that the model might suffer from a problem of underfitting.  


