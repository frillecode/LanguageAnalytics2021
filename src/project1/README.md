# Project 1: Collocates
This project was developed as a solution to assignment 2 set by our teacher, Ross Deans Kristensens-McLachlan, during the course. A Github repository which contains all of the code in relation to my solution to the assignment can be found here: 
https://github.com/frillecode/LanguageAnalytics2021/tree/main/src/project1

## Project description 
### String processing with Python
Using a text corpus found on the cds-language GitHub repo or a corpus of your own found on a site such as Kaggle, write a Python script which calculates collocates for a specific keyword.

- The script should take a directory of text files, a keyword, and a window size (number of words) as input parameters, and an output file called out/{filename}.csv  
- These parameters can be defined in the script itself  
- Find out how often each word collocates with the target across the corpus
- Use this to calculate mutual information between the target word and all collocates across the corpus
- Save result as a single file consisting of three columns: collocate, raw_frequency, MI


## Methods
For this project, I used a dataset consisting of a corpus of 100 English novels, covering the 19th and the beginning of the 20th century. I created a .py-script which takes this corpus, preprocesses the text (remove punctuation, etc.), and calculates collocates for a specific target word. To get a measure of the strength of the association between the target word and the collocates, I calculated the Mutual Information (MI) between them. MI is extracted through calculations of observed and expected frequencies (Evert, 2014):  
MI is given by: _MI = log(O11/E11)_   
, where u is the target word and v is the collocate. 

<p align="center">
    <img src="../../figures/project1_contingencytable.png" width="400" height="200">
  <p>

The window size and target word can be specified through the command-line. 

  
## Usage
The structure of the repository belonging to this assignment is as follows:  
  - Data: _../../data/project1/_ 
  - Code: _collocation.py_
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
$ cd src/project1
$ python3 collocation.py
```

To specify a keyword and window size

You can get more information on which optional arguments can be parsed by running:
``` bash
$ python3 edge_detection.py --help
```

## Discussion of results
The resulting output-files from running the script can be found in 'out/'. 


## References
Evert, S., (2004), "Association Measures", http://collocations.de/AM/index.html, [Retrieved May 21 2021]




