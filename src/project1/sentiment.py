#!/usr/bin/python

# Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import spacy 
nlp = spacy.load("en_core_web_sm") #initialize spaCy
from spacytextblob.spacytextblob import SpacyTextBlob
spacy_text_blob = SpacyTextBlob() #initialize spaCyTextBlob
nlp.add_pipe(spacy_text_blob) #and add it as a new component to our spaCy nlp pipeline


# Defining function for calculating sentiment
def calculate_sentiment(titles):
    polarity = []

    # We use spaCy to create a Doc object for each title. For every doc in this pipe:
    for title in nlp.pipe(titles, batch_size=500): #splitting up into batches and applying to one batch at a time
        # Extract the polarity for each title
        score = title._.sentiment.polarity
        polarity.append(score)  
            
    return polarity


# Defining function for plotting and saving plots
def plotting(x, y, windowsize):
    # create figure
    fig = plt.figure(figsize=(10.0, 3.0)) 
    
    # plot
    plt.plot(x,y, label=f"{windowsize}-days rolling average")

    # naming the x axis 
    plt.xlabel('Publish Date') 
    # naming the y axis 
    plt.ylabel('Polarity') 

    # adding legend
    plt.legend()
    
    # giving a title to my graph 
    plt.title('Daily sentiment score') 

    # function to show the plot 
    plt.show() 
    
    # save plot as .jpg file
    plt.savefig(os.path.join("out", f"sentiment_{windowsize}-days.jpg"))
    plt.close()

    
# Define main-function
def main():
    # Specifying filepath
    in_file = os.path.join("..", "..", "data", "assignment3", "abcnews-date-text.csv")

    # Reading in data
    data = pd.read_csv(in_file) 
    data = data.sample(100000)
    
    # Apply function to calculate sentiment scores and add these to data df
    data["sentiment"] = calculate_sentiment(data["headline_text"])
    
    # Turn publish_date into datetime-object so that Python 'understands' that it is dates
    data["publish_date"] = pd.to_datetime(data["publish_date"], format = "%Y%m%d")
    
    # Calculating average sentiment score per day
    data.index = data['publish_date'] #replace index with "publish_date" column to work with groupby function
    data_average = data.groupby(pd.Grouper(freq='D')).mean() #take daily average of numerical values in df
    data_average = pd.DataFrame.dropna(data_average) #remove row with NAs
    data_average.columns = ["daily_sentiment"]
    
    # Group together polarity scores into windows of 7 and 30 days at a time and calculate an average on that window.
    data_average["smoothed_sentiment_7"] = pd.Series(data_average["daily_sentiment"]).rolling(7).mean()
    data_average["smoothed_sentiment_30"] = pd.Series(data_average["daily_sentiment"]).rolling(30).mean()
    
    # Applying function to create and save plots
    plotting(x = data_average.index, y = data_average["smoothed_sentiment_7"], windowsize = "7")
    plotting(x = data_average.index, y = data_average["smoothed_sentiment_30"], windowsize = "30")

    return print("DONE")

# Define behaviour when called from command line
if __name__=="__main__":
    main()