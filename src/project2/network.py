#!/usr/bin/python

# Load necessary libraries
import os 
import argparse 
import pandas as pd 
import numpy as np
from tqdm import tqdm 
import networkx as nx 
import math 
import holoviews as hv 
from holoviews import opts
from community import community_louvain #(pip install python-louvain)
import matplotlib.pyplot as plt
from collections import Counter 

# Create class for network analysis
class Network_analysis:
    
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(os.path.join("..", "..", "data", "project2", self.args['weighted_edgelist'])) #define data (the weighted edgelist specified in the command line)
        
    def df_fix(self):
        '''This function turns the dataset into a dataframe and filters data according to the cut-off value 
        '''
        # Turn into df
        self.data = pd.DataFrame(self.data, columns=["nodeA", "nodeB", "weight"])

        # Filter data according to cut-off value
        self.filtered_df = self.data[self.data["weight"] > self.args['cutoff']]
    
    def graph(self):
        '''This function creates a graph-object 
        '''
        # Create graph taking nodeA and nodeB and including extra info (here weight)
        self.G = nx.from_pandas_edgelist(self.filtered_df, "nodeA", "nodeB", ["weight"])

    def network_viz(self):
        '''This function takes a graph-object and creates and saves a visualization of the network
        '''
        # Plot 
        pos = nx.random_layout(self.G)
        nx.draw(self.G, pos=pos, with_labels=True, node_size=20, node_color="red",font_size=8,  edge_color="green") #draw plot

        # Save plot
        plot_path = os.path.join("viz", "network.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    def interactive_viz(self):
        '''Creates and saves an interactive network work graph using Holoviews. 
        Edge width is based on weight and node color based on best_partition
        '''
        # Graphic extension
        hv.extension('bokeh')

        # Set layout of graph
        defaults = dict(width=800, height=400)
        hv.opts.defaults(
            opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))

        # Normalize weights for better plotting
        df = self.filtered_df.copy()
        df["weight"] = df["weight"]*len(df)/sum(df["weight"])

        # Compute the best partition to make clusters
        G1 = nx.from_pandas_edgelist(df, source="nodeA", target="nodeB", edge_attr="weight")
        partition = community_louvain.best_partition(G1)

        # Add partitions to the dataframe to get a cluster column and join based on source
        df_partition = pd.DataFrame.from_dict([partition]).transpose().reset_index().rename(columns={0: "cluster"})
        df_partition.columns = ["nodeA", "cluster"]
        df_cluster = pd.merge(df, df_partition, on=["nodeA"])

        # Make the network layout with 
        G = nx.from_pandas_edgelist(df_cluster, source="nodeA", target="nodeB", edge_attr="weight")

        # Add node attributes
        n_A = dict(df_cluster.groupby(["nodeA", "cluster"]).groups.keys())
        n_B = dict(df_cluster.groupby(["nodeB", "cluster"]).groups.keys())

        nx.set_node_attributes(G, n_A, 'cluster')
        nx.set_node_attributes(G, n_B, 'cluster')

        # Create the hv graph
        g = hv.Graph.from_networkx(G, nx.layout.fruchterman_reingold_layout, k=3/math.sqrt(G.order()))

        # Set options for the graph
        g.opts(cmap="spectral", node_size=12, edge_line_width="weight", edge_line_color="grey",
                    node_line_color='gray', node_color = "cluster")
        
        # Add labels
        labels = hv.Labels(g.nodes, ['x', 'y'], 'index')
        g = g*labels.opts(text_font_size= "10pt")

        # Save plot to html
        renderer = hv.renderer('bokeh')
        renderer.save(g, os.path.join("viz", "interactive"))


    def centrality_scores(self):
        '''This function takes a graph-object and calculates and saves degree, betweenness, and eigenvector of nodes
        '''
        #Calculate scores
        bc_metric = nx.betweenness_centrality(self.G) #creates dict with nodes and betweenness scores
        ev_metric = nx.eigenvector_centrality(self.G) #creates dict with nodes and eigenvector scores
        degrees = nx.degree(self.G) #creates iterator over (node, degree) pairs

        # Create df with scores 
        self.centrality_df = pd.DataFrame(bc_metric.items(), columns=["node", "betweenness"]) #create df with nodes and betweenness
        self.centrality_df["eigenvector"] = list(ev_metric.values()) #add eigenvector values as new column
        self.centrality_df["degree"] = [v for k, v in degrees] #extract degrees from (node, degree) pairs and add as new column

        # Save df as CSV
        csv_path = os.path.join("output", "centrality.csv")
        self.centrality_df.to_csv(csv_path, index=False)
    
    def show_top_scores(self):
        '''This function uses the results from the network analysis and prints the top5 in each of the centrality measures 
        '''
        betweenness_top = self.centrality_df.sort_values("betweenness", ascending=False).head(5)
        eigen_top = self.centrality_df.sort_values("eigenvector", ascending=False).head(5)
        degree_top = self.centrality_df.sort_values("degree", ascending=False).head(5)   

        print(f"[INFO] Top 5 entities on betweennes score: \n {betweenness_top}") 
        print(f"[INFO] Top 5 entities on eigenvector score: \n {eigen_top}")    
        print(f"[INFO] Top 5 entities on degree score: \n {degree_top}") 


    def run_analysis(self):
        print("[INFO] loading data")
        self.df_fix()
        print("[INFO] creating graph")
        self.graph()
        print("[INFO] creating visualization")
        self.network_viz()
        print("[INFO] creating interactive visualization")
        self.interactive_viz()
        print("[INFO] calculating centrality-scores")
        self.centrality_scores()
        self.show_top_scores()
  
 
# Define main function
def main():
    ap = argparse.ArgumentParser(description="[INFO] This script takes a weighted edgelist with column names \"nodeA\", \"nodeB\", \"weight\" as input and 1) creates and saves a network visualization and 2) calculates and saves centrality scores. It is possible to specify a cut-off value for which node pairs with edge weight lower than that are removed. This is recommended if the plot is too dense to interpret.")
    # Argument for specifying path to the edgelist you want to use (write -e {name_of_edgelist}.csv in command-line)
    ap.add_argument("-we",  
                    "--weighted_edgelist", 
                    required=False, 
                    default = "weighted_edgelist_test.csv",
                    type=str, 
                    help="str, filename of .csv-file with column headers: nodeA, nodeB, weight (must be in data-folder)")
    # Argument for specifying a cut-off value (write -c {int_value} in command-line)
    ap.add_argument("-c", 
                    "--cutoff", 
                    required=False, 
                    type=int, 
                    default=15, 
                    help="int, cut-off value for filtering node pairs to only include those with edge weight higher than that") 

    args = vars(ap.parse_args())
    
    # Run 
    network = Network_analysis(args)
    network.run_analysis()
    
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()
    print("\n DONE! You can find the network visualizations in 'viz/'and the centrality measures in 'output/' \n")

