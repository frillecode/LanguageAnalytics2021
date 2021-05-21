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

__Bonus challenge__  
- Use argparse to take inputs from the command line as parameters


## Methods
 

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





????????

You can get more information on which optional arguments can be parsed by running:
``` bash
$ python3 edge_detection.py --help
```

## Discussion of results
The resulting output-files from running the script can be found in 'out/'. 




