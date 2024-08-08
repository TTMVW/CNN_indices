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
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens,p_corpus_index[line])#[i])

all_los, all_los_header, all_los_len = load_all_lines(course_all_lo,process_header=True)  
corpus_index = {line[3]:{"institute":line[0],"course":line[1],"level":line[2]}  for line in all_los}

train_corpus = list(read_corpus(courses_train_file,corpus_index)) # ?this makes a list because read_corpus 'yeilds' 
#test_corpus = list(read_corpus(courses_test_file, tokens_only=True))

model = gensim.models.word2vec.Word2Vec.load("bit_lo_model.w2v")
ranks = []
second_ranks = []
doc_sims = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    doc_sims.append((doc_id,sims))
    rank = [docid for docid, sim in sims].index(doc_id)
    
    ranks.append(rank)

    second_ranks.append(sims[1])

print(doc_sims)