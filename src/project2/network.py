#!/usr/bin/python

# Load necessary libraries
import os 
import argparse 
import pandas as pd 
from tqdm import tqdm 
import spacy 
nlp = spacy.load("en_core_web_sm")
import networkx as nx 
import pygraphviz 
import matplotlib.pyplot as plt
from collections import Counter 

# Create class for network analysis
class Network_analysis:
    
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args['weighted_edgelist']) #define data (the weighted edgelist specified in the command line)
        
    def df_fix(self):
        '''This function turns the dataset into a dataframe and filters data according to the cut-off value '''
        # Turn into df
        self.data = pd.DataFrame(self.data, columns=["nodeA", "nodeB", "weight"])

        # Filter data according to cut-off value
        self.filtered_df = self.data[self.data["weight"] > self.args['cutoff']]
    
    def graph(self):
        '''This function creates a graph-object '''
        # Create graph taking nodeA and nodeB and including extra info (here weight)
        self.G = nx.from_pandas_edgelist(self.filtered_df, "nodeA", "nodeB", ["weight"])

    def network_viz(self):
        '''This function takes a graph-object and creates and saves a visualization of the network'''
        # Plot 
        pos = nx.nx_agraph.graphviz_layout(self.G, prog="neato") #creates position for plot
        nx.draw(self.G, pos, with_labels=True, node_size=20, font_size=5) #draw plot

        # Save plot
        plot_path = os.path.join("viz", "network.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")


    def centrality_scores(self):
        '''This function takes a graph-object and calculates and saves degree, betweenness, and eigenvector of nodes'''
        #Calculate scores
        bc_metric = nx.betweenness_centrality(self.G) #creates dict with nodes and betweenness scores
        ev_metric = nx.eigenvector_centrality(self.G) #creates dict with nodes and eigenvector scores
        degrees = nx.degree(self.G) #creates iterator over (node, degree) pairs

        # Create df with scores 
        centrality_df = pd.DataFrame(bc_metric.items(), columns=["node", "betweenness"]) #create df with nodes and betweenness
        centrality_df["eigenvector"] = list(ev_metric.values()) #add eigenvector values as new column
        centrality_df["degree"] = [v for k, v in degrees] #extract degrees from (node, degree) pairs and add as new column

        # Save df as CSV
        csv_path = os.path.join("output", "centrality.csv")
        centrality_df.to_csv(csv_path)
        
    def run_analysis(self):
        self.df_fix()
        self.graph()
        self.network_viz()
        self.centrality_scores()
  
 
# Define main function
def main():
    ap = argparse.ArgumentParser(description="[INFO] This script takes a weighted edgelist with column names \"nodeA\", \"nodeB\", \"weight\" as input and 1) creates and saves a network visualization and 2) calculates and saves centrality scores. It is possible to specify a cut-off value for which node pairs with edge weight lower than that are removed. This is recommended if the plot is too dense to interpret.")
    # Argument for specifying path to the edgelist you want to use (write -e {name_of_edgelist}.csv in command-line)
    ap.add_argument("-we",  
                    "--weighted_edgelist", 
                    required=True, 
                    type=argparse.FileType('r'), 
                    help="dataframe, path to .csv-file with column headers: nodeA, nodeB, weight")
    # Argument for specifying a cut-off value (write -c {int_value} in command-line)
    ap.add_argument("-c", 
                    "--cutoff", 
                    required=False, 
                    type=int, 
                    default=0, 
                    help="int, cut-off value for filtering node pairs to only include those with edge weight higher than that") 

    args = vars(ap.parse_args())
    
    # Run 
    network = Network_analysis(args)
    network.run_analysis()
    
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()
    print("\n DONE! You can find the network visualization in \'viz\'and the centrality measures in \'output\' \n")