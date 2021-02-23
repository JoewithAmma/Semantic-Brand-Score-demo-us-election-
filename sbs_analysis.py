import csv
f = open("newsdata.txt", "r",encoding='UTF-8')
txt = f.read()
f.close()
texts = [txt]
#4 Chapters of Alice in Wonderland
print(len(texts))
#print(texts)
#print(texts[0][:200])

import re
import nltk
import string
from nltk.stem.snowball import SnowballStemmer

#Define stopwords
#nltk.download("stopwords")
stopw = nltk.corpus.stopwords.words('english') + ['‘']

#Define brands (lowercase)
brands = ['trump', 'biden']

# texts is a list of strings, one for each document analyzed.

#Convert to lowercase
texts = [t.lower() for t in texts]
#Remove words that start with HTTP
texts = [re.sub(r"http\S+", " ", t) for t in texts]
#Remove words that start with WWW
texts = [re.sub(r"www\S+", " ", t) for t in texts]
#Remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
texts = [regex.sub(' ', t) for t in texts]
#Remove words made of single letters
texts = [re.sub(r'\b\w{1}\b', ' ', t) for t in texts]
#Remove stopwords
pattern = re.compile(r'\b(' + r'|'.join(stopw) + r')\b\s*')
texts = [pattern.sub(' ', t) for t in texts]
#Remove additional whitespaces
texts = [re.sub(' +',' ',t) for t in texts]

#Tokenize text documents (becomes a list of lists)
texts = [t.split() for t in texts]

# Snowball Stemming
stemmer = SnowballStemmer("english")
texts = [[stemmer.stem(w) if w not in brands else w for w in t] for t in texts]
#texts[0][:6]
#print(texts)

from collections import Counter
import numpy as np

#Create a dictionary with frequency counts for each word
countPR = Counter()
for t in texts:
    countPR.update(Counter(t))

#Calculate average score and standard deviation
avgPR = np.mean(list(countPR.values()))
stdPR = np.std(list(countPR.values()))

#Calculate standardized Prevalence for each brand
PREVALENCE = {}
for brand in brands:
    PREVALENCE[brand] = (countPR[brand] - avgPR) / stdPR
    print("Prevalence", brand, PREVALENCE[brand])
    
import networkx as nx

#Choose a co-occurrence range
co_range = 7

#Create an undirected Network Graph
G = nx.Graph()

#Each word is a network node
nodes = set([item for sublist in texts for item in sublist])
G.add_nodes_from(nodes)

#Add links based on co-occurrences
for doc in texts:
    w_list = []
    length= len(doc)
    for k, w in enumerate(doc):
        #Define range, based on document length
        if (k+co_range) >= length:
            superior = length
        else:
            superior = k+co_range+1
        #Create the list of co-occurring words
        if k < length-1:
            for i in range(k+1,superior):
                linked_word = doc[i].split()
                w_list = w_list + linked_word
        #If the list is not empty, create the network links
        if w_list:    
            for p in w_list:
                if G.has_edge(w,p):
                    G[w][p]['weight'] += 1
                else:
                    G.add_edge(w, p, weight=1)
        w_list = []

#Remove negligible co-occurrences based on a filter
link_filter = 2
#Create a new Graph which has only links above
#the minimum co-occurrence threshold
G_filtered = nx.Graph() 
G_filtered.add_nodes_from(G)
for u,v,data in G.edges(data=True):
    if data['weight'] >= link_filter:
        G_filtered.add_edge(u, v, weight=data['weight'])

#Optional removal of isolates
isolates = set(nx.isolates(G_filtered))
isolates -= set(brands)
G_filtered.remove_nodes_from(isolates)

#Check the resulting graph (for small test graphs)
#G_filtered.nodes()
#G_filtered.edges(data = True)
print("Original Network\nNo. of Nodes:", G.number_of_nodes(), "No. of Edges:", G.number_of_edges())
print("Filtered Network\nNo. of Nodes:", G_filtered.number_of_nodes(), "No. of Edges:", G_filtered.number_of_edges())


from distinctiveness.dc import distinctiveness
import networkx as nx
#DIVERSITY
#Calculate Distinctiveness Centrality
DC = distinctiveness(G_filtered, normalize = False, alpha = 1)
DIVERSITY_sequence=DC["D2"]

#Calculate average score and standard deviation
avgDI = np.mean(list(DIVERSITY_sequence.values()))
stdDI = np.std(list(DIVERSITY_sequence.values()))
#Calculate standardized Diversity for each brand
DIVERSITY = {}
for brand in brands:
    DIVERSITY[brand] = (DIVERSITY_sequence[brand] - avgDI) / stdDI
    print("Diversity", brand, DIVERSITY[brand])
    
for u,v,data in G_filtered.edges(data=True):
    if 'weight' in data and data['weight'] != 0:
        data['inverse'] = 1/data['weight']
    else:
        data['inverse'] = 1   



#CONNECTIVITY
CONNECTIVITY_sequence=nx.betweenness_centrality(G_filtered, normalized=False, weight ='inverse')
#Calculate average score and standard deviation
avgCO = np.mean(list(CONNECTIVITY_sequence.values()))
stdCO = np.std(list(CONNECTIVITY_sequence.values()))
#Calculate standardized Prevalence for each brand
CONNECTIVITY = {}
for brand in brands:
    CONNECTIVITY[brand] = (CONNECTIVITY_sequence[brand] - avgCO) / stdCO
    print("Connectivity", brand, CONNECTIVITY[brand])
    
SBS = {}
for brand in brands:
    SBS[brand] = PREVALENCE[brand] + DIVERSITY[brand] + CONNECTIVITY[brand]
    print("SBS", brand, SBS[brand])
    
import pandas as pd

PREVALENCE = pd.DataFrame.from_dict(PREVALENCE, orient="index", columns = ["PREVALENCE"])
DIVERSITY = pd.DataFrame.from_dict(DIVERSITY, orient="index", columns = ["DIVERSITY"])
CONNECTIVITY = pd.DataFrame.from_dict(CONNECTIVITY, orient="index", columns = ["CONNECTIVITY"])
SBS = pd.DataFrame.from_dict(SBS, orient="index", columns = ["SBS"])

SBS = pd.concat([PREVALENCE, DIVERSITY, CONNECTIVITY, SBS], axis=1, sort=False)
