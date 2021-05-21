
# Project 4: Text classification using Deep Learning
This project was developed as a solution to assignment 6 set by our teacher, Ross Deans Kristensens-McLachlan, during the course. A Github repository which contains all of the code in relation to my solution to the assignment can be found here: 
https://github.com/frillecode/LanguageAnalytics2021/tree/main/src/project5

## Project description 
### Text classification using Deep Learning
In class this week, we've seen how deep learning models like CNNs can be used for text classification purposes. For your assignment this week, I want you to see how successfully you can use these kind of models to classify a specific kind of cultural data - scripts from the TV series Game of Thrones.

You can find the data here: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons

In particular, I want you to see how accurately you can model the relationship between each season and the lines spoken. That is to say - can you predict which season a line comes from? Or to phrase that another way, is dialogue a good predictor of season?

Start by making a baseline using a 'classical' ML solution such as CountVectorization + LogisticRegression and use this as a means of evaluating how well your model performs. Then you should try to come up with a solution which uses a DL model, such as the CNNs we went over in class.


## Methods
 

## Usage
The structure of the files belonging to this assignment is as follows:
  - Data: See README in _../../data/project4/_ for instructions on getting the data
  - Code: _LR\_GOT.py_  ,  _DL\_GOT.py_
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
For this assignment, I used pretrained word embeddings from ```GloVe```. To run the script, these needs to be downloaded and placed in the data-folder. This can be done by running the following in the command-line:
```bash
$ cd CDS-spring-2021-language/data/project4/glove
$ wget http://nlp.stanford.edu/data/glove.6B.zip
$ unzip -q glove.6B.zip
```

After updating the repo (see above), you can run the .py-files from the command-line by writing the following:
``` bash
$ cd LanguageAnalytics2021
$ source cds-lang/bin/activate
$ cd src/project5
$ python3 LR_GOT.py
$ python3 DL_GOT.py
```





????????

You can get more information on which optional arguments can be parsed by running:
``` bash
$ python3 LR_GOT.py --help
```

## Discussion of results
The resulting output-files from running the script can be found in 'out/'. 


For this assignment, I analysed a [dataset](https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons) consisting of scripts from all seasons of the TV-series, _Game of Thrones_. I want to investigate how accurately I can model the relationship between each season train and the lines spoken using Deep Learning (DL). To do so, I want to create a Convolutional Neural Network model which attempt to predict which season a sentence comes from. 

To investigate how well I could train a DL model to predict the season based on sentences, I first created a logistic regression (LR) classification model to use as a baseline. I trained the LR model to predict the season of sentences based on a count-vector of the words in the sentences. Because the data was sparse, I scaled the data using the inbuilt MaxAbsScaler() from sklearn (dividing by max value for each feature to get values between -1 and 1), which is a solution suggested in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/preprocessing.html). The amount of data for each season varied (with Season 2 having 3914 sentences and Season 6 having only  1466 sentences). To overcome this, I created a more balanced dataset based on samples from each season. Because of the large difference in number of sentences between seasons, the balancing results in a exclusion of a lot of data. When comparing the model with balanced data to the model with the original data, we see that the balanced data results in a worse performance of the model. I, therefore, decided to use the LR model trained on the original (unabalanced) dataset as the baseline model to compare againts the DL model. You can find a confusion matrix and classifier metrics for this model in the _out_-folder. The LR model performed with an accuracy of 0.26 with f1-scores ranging between 0.13 and 0.31 indicating that the model differed quite a lot in how succesful it predicted the different seasons. Especially season 8 and 6 seemed to cause issues. To validate the results, I performed cross-validation to get average performance using different test-train splits. The results of this can also be found in the _out_-folder. 


The DL model...


