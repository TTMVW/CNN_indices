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
courses_train_file = os.path.join(test_data_dir, 'training.cor')
courses_test_file = os.path.join(test_data_dir, 'test.cor')



def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(courses_train_file))
test_corpus = list(read_corpus(courses_test_file, tokens_only=True))

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
    
#print(doc_sims)