import os
import smart_open
import gensim
import networkx as nx
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
from keras.models import Model
import random
import time
from itertools import combinations
import pulp
from pulp import LpBinary, LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum



test_data_dir = os.getcwd() # os.path.join(gensim.__path__[0], 'test', 'test_data')
courses_train_file = os.path.join(test_data_dir, 'training.cor') #training corpora is actually all randomised
courses_test_file = os.path.join(test_data_dir, 'test.cor')
course_all_lo = os.path.dirname(test_data_dir)+"/all_lo.csv"

def load_all_lines(p_path_filename, process_header=False):
    #f = open(p_path_filename, "r")
    f = smart_open.open(p_path_filename, encoding="iso-8859-1")
    lines = f.readlines()
    def tuplelise_lo_row(p_line :str ):
        quote_line =  p_line.strip().split('"')
        if len(quote_line) > 1 :
            front = quote_line[0].split(",")[:-1]
            lo_str = quote_line[1]
            whole = front + [lo_str]
        else: 
            whole = ["__ERR__"]+ quote_line 
        whole = tuple(whole)
        return whole
            
    
    lines = [ tuplelise_lo_row(aline[:-1]) for aline in lines] # get rid of "/n" s
    result = []
    header = []
    if process_header:
        result = lines[1:]
        header.append(lines[0])
    else:
        result = lines
            
    return result, header, len(result)

def read_corpus(fname, p_corpus_index,tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            line = line[:-1]
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags from corpus index
                yield gensim.models.doc2vec.TaggedDocument(tokens,p_corpus_index[line])#[i])

def make_big_graph(doc_sims,threshold):  
    def tuple_tag(tag):
        result = ()
        for key in tag:
            result+=tuple(tag[key])
        return result
            
    the_big_graph = nx.Graph()
    graph_nodes = []
    graph_edges = []
    # for i in range(0,len(doc_sims)):
    #     first = doc_sims[i][0]
    #     index = first[0]
    #     tag = tuple_tag(first[1])
    #     graph_nodes += [(index, tag)]
        
        
    # the_big_graph.add_nodes_from(graph_nodes)
        
    added = []   
    for i in range(0,len(doc_sims)):
        first = doc_sims[i][0]
        index = first[0] 
        tag = tuple_tag(first[1])   
        the_sims = doc_sims[0][1]
        # print((index,tag))
        for a_sim in the_sims:
            similarity = a_sim[2]
            #print(similarity)
            if similarity > threshold:
                sim_index = a_sim[0]
                sim_tag = tuple(a_sim[1])
                if ( (sim_index,index) not in added) \
                        and  ((index,sim_index) not in added):
                    graph_edges += [(index,tag),(sim_index,sim_tag)]
                    added += [(index,sim_index),(sim_index,index)]
                
                
                
    the_big_graph.add_edges_from(graph_edges)
                
    return the_big_graph

all_los, all_los_header, all_los_len = load_all_lines(course_all_lo,process_header=True)  
corpus_index = {line[3]:{"institute":line[0],"course":line[1],"level":line[2]}  for line in all_los}

train_corpus = list(read_corpus(courses_train_file,corpus_index)) # ?this makes a list because read_corpus 'yeilds' 
#test_corpus = list(read_corpus(courses_test_file, tokens_only=True))

model = gensim.models.word2vec.Word2Vec.load("bit_lo_model.w2v")
# ranks = []
# second_ranks = []
doc_sims = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    tagged_sims = [(asim[0],train_corpus[asim[0]].tags,asim[1]) for asim in sims if asim[0] != doc_id] 
    doc_sims.append(((doc_id,train_corpus[doc_id].tags),tagged_sims))
    # rank = [docid for docid, sim in sims].index(doc_id)
    
    # ranks.append(rank)

    # second_ranks.append(sims[1])
    

            
#the_big_graph = make_big_graph(doc_sims,0.7)
the_big_graph = nx.Graph()
edges = []
nodes = []
number_of_docs = len(doc_sims)


threshold = 0.8
for i in range(0,number_of_docs):
    current_sim = doc_sims[i]
    sims = current_sim[1]
    sims_above_threshold = [ sim[0] for sim in sims if sim[2] >= threshold]
    if len(sims_above_threshold) > 0:
        for j in range(i+1, number_of_docs):
            if j in sims_above_threshold:
                edges += [[i,j]]
    nodes += [i]
        
       
the_big_graph.add_edges_from(edges)  
the_big_graph.add_nodes_from(nodes)     
print("Number of docs:",number_of_docs,"Graph:",the_big_graph)