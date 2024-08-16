import os
import smart_open
import gensim
import networkx as nx
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from PIL import Image
import matplotlib as mpl
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
# mpl.use('TkAgg')

import datetime


test_data_dir = os.getcwd() # os.path.join(gensim.__path__[0], 'test', 'test_data')
courses_train_file = os.path.join(test_data_dir, 'training.cor') #training corpora is actually all randomised
courses_test_file = os.path.join(test_data_dir, 'test.cor')
course_all_lo =  os.path.join(test_data_dir, 'all_lo_bit_org.csv')
corpus_to_check = os.path.join(test_data_dir, 'HUT_course_los_all_2024.csv')

def write_to_file(filename, content):
    # Open the file in write mode ('w')
    # If the file doesn't exist, it will be created
    with open(filename, 'w') as file:
        # Write the content to the file
        file.write(content)
        
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
                

all_los, all_los_header, all_los_len = load_all_lines(course_all_lo,process_header=True)  
corpus_index = {line[3]:{"institute":line[0],"course":line[1],"level":line[2]}  for line in all_los}

train_corpus = list(read_corpus(courses_train_file,corpus_index)) # ?this makes a list because read_corpus 'yeilds' 
#test_corpus = list(read_corpus(courses_test_file, tokens_only=True))

model = gensim.models.word2vec.Word2Vec.load("bit_lo_model.w2v")



def get_graph_doc_sims(train_corpus):
    # ranks = []
    # second_ranks = []
    doc_sims = []
    similarity_matrix = [] 
    sim_matrix_start = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        tagged_sims = [(asim[0],train_corpus[asim[0]].tags,asim[1]) for asim in sims if asim[0] != doc_id] 
        tagged_sims = [(doc_id,train_corpus[doc_id].tags,1)] + tagged_sims
        sorted_tagged_sims = sorted(tagged_sims, key=lambda x: x[0])
        sim_row = []
        for i in range(0,len(train_corpus)):
            if i != doc_id:
                sim_row+= [sorted_tagged_sims[i][2]]
            else:
                sim_row+=[1]
        sim_matrix_start  += [sim_row]
        doc_sims.append(((doc_id,train_corpus[doc_id].tags),sorted_tagged_sims))
        
    similarity_matrix = np.array(sim_matrix_start)  
    
    return doc_sims, similarity_matrix

def doc_sims_all_csv(pDocSims):
    def details_to_csv(pDetails):
        
        return f"{pDetails['institute']},{pDetails['course']},{pDetails['level']}"
    
    lines = ""
    for i in range(0, len(pDocSims)):
       
        front_csv = details_to_csv(pDocSims[i][0][1]) 
        
        a_line = front_csv
        sims = pDocSims[i][1]
        for j in range(0, len(sims)):
             sim_front = details_to_csv(sims[j][1])
             simularity = str(sims[j][2])
             a_line += ","+sim_front+","+simularity+"\n"
             lines += a_line
             a_line = front_csv
             
    write_to_file("./All_Course_Similarities.csv",lines)       
            
        
            
    


        
# for NMIT run LO simliarity with the HUT LOs
# def get_docs_similar_los(the_corpora_list,threshold, train_corpus):
#     for a_corpora in the_corpora_list:
#         LOs = a_corpora[3]
#         course_name = a_corpora[1]
#         level = a_corpora[2]
    
#         tokens = gensim.utils.simple_preprocess(LOs)
#         inferred_vector = model.infer_vector(tokens)
#         sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
#         count = 0
#         printed = False
#         for a_sim in sims:
            
#             similarity = a_sim[1]
#             doc_id = a_sim[0]
#             if similarity > threshold:
#                 tags = train_corpus[doc_id].tags
#                 if tags["institute"] == "NMIT" :
#                     print( f"{course_name},{level},{similarity},{tags['institute']},{tags['course']},{tags['level']}")
#                     printed = True
            
#         if not printed:
#             print( f"{course_name},{level},,,,")
            
            
# Corpus of documents each line is "document name","document text"                
# hut_tuples,_,_ =  load_all_lines(corpus_to_check, process_header=True)  
# get_docs_similar_los(hut_tuples,0.4,train_corpus)   
 

# ====== START GRAPHS =========   
doc_sims, similarity_matrix =  get_graph_doc_sims(train_corpus)
doc_sims_all_csv(doc_sims)

# # similarity_matrix = np.array(sim_matrix_start)  
# print("Made similarity matrix")
# # Create a graph snip
# G = nx.Graph()
# L5 = nx.Graph()
# L6 = nx.Graph()
# L7 = nx.Graph()
# # Add nodes and edges with weights (similarities)
# threshold = 0.8
# institute_to_institute_list = {}
# institute_list = {}
# institute_level_list = {}


# for i in range(similarity_matrix.shape[0]):
#     for j in range(i + 1, similarity_matrix.shape[0]):
#         if similarity_matrix[i, j] > threshold:
#             G.add_edge(i, j, weight=similarity_matrix[i, j])
            
#             # Graphs by level
#             from_level = doc_sims[i][0][1]['level']
#             to_level = doc_sims[j][0][1]['level'] 
#             if from_level == to_level:
#                 match from_level:
#                     case '5': 
#                         L5.add_edge(i, j, weight=similarity_matrix[i, j]) 
#                     case '6':
#                         L6.add_edge(i, j, weight=similarity_matrix[i, j]) 
#                     case '7':
#                         L7.add_edge(i, j, weight=similarity_matrix[i, j]) 
#                     case _:
#                         pass
#             # Graphs by institute
#             from_institute = doc_sims[i][0][1]['institute']
#             to_institute = doc_sims[j][0][1]['institute']
            
#             if from_institute == to_institute:
#                 if from_institute not in institute_list:
#                     institute_list[from_institute] = nx.Graph()
#                 institute_list[from_institute].add_edge(i, j, weight=similarity_matrix[i, j]) 
#                 from_level = doc_sims[i][0][1]['level']
#                 to_level = doc_sims[j][0][1]['level']
#                 from_institute_level = from_institute + "_level_"+from_level
#                 if from_level == to_level :
#                     if from_institute_level not in institute_level_list:
#                         institute_level_list[from_institute_level] = nx.Graph()
#                     institute_level_list[from_institute_level].add_edge(i, j, weight=similarity_matrix[i, j]) 
#             else:
#                 from_to = from_institute+"_to_all"
#                 if  from_to not in institute_to_institute_list:
#                     institute_to_institute_list[from_to] = nx.Graph()
#                 institute_to_institute_list[from_to].add_edge(i, j, weight=similarity_matrix[i, j])
                            
                
            

        
# print("Made graph")
# print(G)
# print(L5)
# print(L6)
# print(L7)


    

# # # Draw the graph using a spring layout (force-directed)

# def draw_force_directed(pG,pName:str,big=False):
#     print("Drawing ..", pName, pG)
#     current_time = datetime.datetime.now()
#     formatted_time = current_time.strftime('%H:%M:%S')
#     print(formatted_time)
#     plt.cla()
#     #fig = plt.figure(1, figsize=(200, 200), dpi=60)
#     print("Starting Draw")
#     if big:
#         pos = nx.spring_layout(pG)
#         nx.draw(pG, pos, with_labels=True, node_size=1 , node_shape=",",node_color='lightblue', edge_color='black')
#         nx.draw_networkx_edge_labels(pG, pos, edge_labels={(i, j): f'{similarity_matrix[i, j]:.2f}' for i, j in pG.edges()})
#     else:
#         nx.draw(pG, with_labels=True, node_size=10 , node_shape=",",node_color='lightblue', edge_color='black')
#     formatted_time = current_time.strftime('%H:%M:%S')
#     print(formatted_time)
#     ##pos = nx.spring_layout(pG)
#     #pos = nx.kamada_kawai_layout(pG)
#     #width_values = [5 for i in range(0,nx.number_of_edges(pG))]
#     # if nx.number_of_nodes(pG) > 10:
#     #     nx.draw(pG, pos, with_labels=True, node_size=10 , node_shape=",",node_color='lightblue', edge_color='black')
#     #     nx.draw_networkx_edges(pG,pos, width=width_values)
#     #     nx.draw_networkx_edge_labels(pG, pos, edge_labels={(i, j): f'{similarity_matrix[i, j]:.2f}' for i, j in pG.edges()})
#     # else:
#     #     nx.draw(pG, with_labels=True, node_size=10 , node_shape=",",node_color='lightblue', edge_color='black')
#         #nx.draw_networkx_edge_labels(pG, edge_labels={(i, j): f'{similarity_matrix[i, j]:.2f}' for i, j in pG.edges()})
    
        
#     print(f"About to save fig {pName}")
#     plt.savefig(f"./latest_graphs/{pName}_similarity.png")


# fig = plt.figure(1, figsize=(200, 200), dpi=60)  
# draw_force_directed(L5,'Level_5',big=True) 
# draw_force_directed(L6,'Level_6',big=True)
# draw_force_directed(L7,'Level_7',big=True)     

# for k,g in institute_list.items():
#     print(g)  
#     draw_force_directed(g,k,big=True)    

# count = 0  
# for k,g in institute_level_list.items():
#      draw_force_directed(g,k,big=True)
     
# for k,g in institute_to_institute_list.items():
#      draw_force_directed(g,k,big=True)
     

# draw_force_directed(G,'All_courses', big=True) 

# # # #the_big_graph = make_big_graph(doc_sims,0.7)
# # # the_big_graph = nx.Graph()
# # # edges = []
# # # nodes = []
# # # number_of_docs = len(doc_sims)


# # # threshold = 0.8
# # # for i in range(0,number_of_docs):
# # #     current_sim = doc_sims[i]
# # #     sims = current_sim[1]
# # #     sims_above_threshold = [ sim[0] for sim in sims if sim[2] >= threshold]
# # #     if len(sims_above_threshold) > 0:
# # #         for j in range(i+1, number_of_docs):
# # #             if j in sims_above_threshold:
# # #                 edges += [[i,j]]
# # #     nodes += [i]
        
       
# # # the_big_graph.add_edges_from(edges)  
# # # the_big_graph.add_nodes_from(nodes)     
# # # print("Number of docs:",number_of_docs,"Graph:",the_big_graph)
# # # print("Starting layout")
# # # pos = nx.kamada_kawai_layout(the_big_graph,scale=10.0)
# # # #pos = nx.spring_layout(the_big_graph)
# # # #pos = nx.shell_layout(the_big_graph)
# # # print("Drawing")
# # # nx.draw(the_big_graph,pos, node_size=1, node_shape=",")

# # # plt.savefig("./the_big_graph2.png")
# # # print("Finished All Nodes")

# # # print("Starting level 5 -7")
# # # level5_nodes= []
# # # level6_nodes = []
# # # level7_nodes = []
# # # def level_node(level_nodes,n,_from):
# # #     if train_corpus[_from][1]['level'] == str(n) :
# # #         level_nodes += [_from]
# # #     return level_nodes
    
# # # for _from in the_big_graph.nodes:
# # #     level5_nodes = level_node(level5_nodes,5,_from)
# # #     level6_nodes = level_node(level6_nodes,6,_from)
# # #     level7_nodes = level_node(level7_nodes,7,_from)
# # # level5_subgraph = the_big_graph.subgraph(level5_nodes)    
# # # level6_subgraph = the_big_graph.subgraph(level6_nodes)
# # # level7_subgraph = the_big_graph.subgraph(level7_nodes)

# # # def draw_level(sub_graph,pict_name):
# # #     pos = nx.kamada_kawai_layout(level5_subgraph,scale=10.0)
# # #     nx.draw(sub_graph,pos, node_size=1, node_shape=",")
# # #     plt.savefig(f"./{pict_name}.png")
    
# # # print("Drawing level 5")
# # # print(level5_subgraph)
# # # draw_level(level5_subgraph,"level5.1")

# # # print("Drawing level 6")
# # # print(level6_subgraph)
# # # draw_level(level6_subgraph,"level6.1")


# # # print("Drawing level 7")
# # # print(level7_subgraph)
# # # draw_level(level7_subgraph,"level7.1")

        
# # #plt.show()
