#!/usr/bin/python

# Import necessary libraries
import os
import re
import string
import numpy as np
from os import listdir
from pathlib import Path
import pandas as pd
from collections import Counter


# Define function for calculating MI
def collocate_function(filepath, keyword, window_size):
    
    all_texts = []
    
    # Load files from directory
    for filename in Path(filepath).glob("*.txt"):
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()
            all_texts.append(text)

    corpus = " ".join(all_texts)
    
    corpus = re.sub(r"\W+", " ", corpus) #remove punctuation
    corpus = corpus.lower() #lowercase

    tokenized = [token for token in corpus.split()] #tokenizing - splitting by whitespaces

    counter_object = Counter(tokenized) #returns number of time each element, i.e. collocate, appears in list      

    u = counter_object.get(keyword)

    collocates = []
    
    for word_n in range(len(tokenized)):

        if tokenized[word_n] == keyword:

            index_value = word_n

            left_window = max(0, index_value - window_size) #2 words on left side
            right_window = index_value + window_size + 1 #2 words on right side

            window_list = tokenized[left_window : right_window]

            for word in window_list:
                if word == keyword:
                    pass
                else:
                    collocates.append(word) #save word             

    collocate = [x for x in Counter(collocates).keys()] #extracting collocates from dictionary to a list

    O11 = [x for x in Counter(collocates).values()] #n times collcoate 

    O12 = [x1 - x2 for (x1, x2) in zip(([u] * len(O11)), O11)]

    R1 = [x1 + x2 for (x1, x2) in zip(O11, O12)]

    C1 = [counter_object.get(w) for w in collocate] #raw frequency



    # length of text
    N = len(tokenized)

    # Expected
    E11 = [x1 * x2 for (x1, x2) in zip(R1, C1)]
    E11 = [x1 / x2 for (x1, x2) in zip(E11, ([N]*len(E11)))]

    # return MI
    MI = [np.log(x1/x2) for (x1, x2) in zip(O11, E11)]

    df = pd.DataFrame()
    df["collocate"] = collocate
    df["raw_frequency"] = C1
    df["MI"] = MI
    
    return df

# Define main function
def main():
    # Run collocate-function (here I use the keyword "father" and a windowsize of 2)
    df = collocate_function(filepath = os.path.join("..", "..", "data", "assignment1+2", "100_english_novels", "corpus"), 
                       keyword = "father", 
                       window_size = 2)
    
    # Save file
    outpath = os.path.join("out", "collocates.csv")
    df.to_csv(outpath)
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()