#!/usr/bin/python

# Import necessary libraries
import os
import re
import string
import argparse
import numpy as np
from os import listdir
from pathlib import Path
import pandas as pd
from collections import Counter


class Collocation:
    """A class for calculating Mutual Information (MI) with collocates given a target word and a window size. The class contains functions for loading and preprocessing data, extracting relevant information, and calculating MI. A dataframe with collocate, raw_frequency, and MI is returned. 
    """
    def __init__(self, args):
        self.args = args
       
        
    # Define function for calculating MI
    def preprocessing(self, filepath):
        """Function for loading in text data from folder, creating a corpus of all texts, and preprocessing it. 
        Input:
            filepath: str, path to folder with text data
        """
        all_texts = []

        # Load files from directory
        for filename in Path(filepath).glob("*.txt"):
            with open(filename, "r", encoding="utf-8") as file:
                text = file.read()
                all_texts.append(text)
                
        # Create corpus from all texts
        self.corpus = " ".join(all_texts)
        
        # Data cleaning
        self.corpus = re.sub(r"\W+", " ", self.corpus) #remove punctuation
        self.corpus = self.corpus.lower() #lowercase
        self.tokenized = [token for token in self.corpus.split()] #tokenizing - splitting by whitespaces

    def collocations(self):
        """Function for calculating collocations given target word and a window size
        """

        # Looping through all occurences of target word in the corpus
        self.collocates = []
        for word_n in range(len(self.tokenized)): 
            # When target word appears, use index value to create window
            if self.tokenized[word_n] == self.args['keyword']:
                index_value = word_n #save index value of target word
                left_window = max(0, index_value - self.args['window_size']) #2 words on left side
                right_window = index_value + self.args['window_size'] + 1 #2 words on right side
                
                # Create list of collocates for this occurence of the target word
                window_list = self.tokenized[left_window : right_window]
                
                # Save words into collocate-list for all occurences of target word
                for word in window_list:
                    if word == self.args['keyword']: #ensure that target word does not appear with itself
                        pass
                    else:
                        self.collocates.append(word) #save word   
                        
        self.collocate = [x for x in Counter(self.collocates).keys()] #extracting collocates from dictionary to a list


    def mutual_information(self):
        """This function calculates Mutual Information between a target word and its collocates
        """
        
        # Calculating observed frequencies
        counter_object = Counter(self.tokenized) #returns number of time each element appears in list
        
        u = counter_object.get(self.args['keyword']) #extracting count of target word
        
        O11 = [x for x in Counter(self.collocates).values()] #occurence of target word with collocate

        O12 = [x1 - x2 for (x1, x2) in zip(([u] * len(O11)), O11)] #occurence of target word without collocate

        R1 = [x1 + x2 for (x1, x2) in zip(O11, O12)] #raw frequency of target word
        
        C1 = [counter_object.get(w) for w in self.collocate] #raw frequency of collocate

        N = len(self.tokenized) #length of text

        # Calculating expected frequencies
        E11 = [x1 * x2 for (x1, x2) in zip(R1, C1)] 
        E11 = [x1 / x2 for (x1, x2) in zip(E11, ([N]*len(E11)))] 

        # Calculate MI based on expected and observed co-occurences
        MI = [np.log(x1/x2) for (x1, x2) in zip(O11, E11)]

        # Save information in dataframe
        df = pd.DataFrame({"collocate": self.collocate,
                           "raw_frequency": C1,
                           "MI": MI})
        return df

# Define main function
def main():
    ap = argparse.ArgumentParser(description="[INFO] This script takes text data and calculates Mutual Information scores between collocates and a given target word. The window size can be defined.")
    # Argument for specifying pixel values for resizing images
    ap.add_argument("-w", 
                "--window_size", 
                required=False, 
                type=int,
                default=2,
                help="int, window size for co-occurence (e.g 2 means 2 words before + 2 words after target word)") 
    # Argument for specifying target word
    ap.add_argument("-k",  
                "--keyword", 
                required=False, 
                type=str, 
                default="father", 
                help="str, target word")

    
    args = vars(ap.parse_args())
    
    
    # Runnning analysis
    col = Collocation(args) #class object
    col.preprocessing(filepath = os.path.join("..", "..", "data", "project1", "100_english_novels", "corpus")) #preprocessing
    col.collocations() #calculating frequencies
    df = col.mutual_information() #calculating MI
    
    # Save file
    outpath = os.path.join("out", "collocates.csv")
    df.to_csv(outpath, index=False)
    
    # Top 10
    top10 = df[df["raw_frequency"] > 1].sort_values("MI", ascending=False).head(10)

    print(f"[INFO] DONE! You can now find the complete results in the 'out/'-folder. The top 10 of words (raw_freq>1) with highest MI are: \n {top10}")

# Define behaviour when called from command line
if __name__=="__main__":
    print(f"[INFO] Calculating....")
    main()

