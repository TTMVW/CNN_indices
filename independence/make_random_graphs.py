import networkx as nx
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import random
import os
import shutil
import subprocess

def run_bash_command(command):
    try:
        # Run the bash command using subprocess.run
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Print the command's output
        print("Output:\n", result.stdout.decode())
        
        # Print the command's error output if any
        if result.stderr:
            print("Error:\n", result.stderr.decode())
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error:\n{e.stderr.decode()}")


def write_to_file(filename, content):
    # Open the file in write mode ('w')
    # If the file doesn't exist, it will be created
    with open(filename, 'w') as file:
        # Write the content to the file
        file.write(content)
        
def make_run_folder(folder_name,file_to_copy):
    # Get the current working directory
    current_directory = os.getcwd()

    # Create the new folder path
    new_folder_path = os.path.join(current_directory, folder_name)

    # Create the new folder if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Define the source file path and destination path
    source_file_path = os.path.join(current_directory, file_to_copy)
    destination_file_path = os.path.join(new_folder_path, file_to_copy)

    # Copy the file
    shutil.copy2(source_file_path, destination_file_path)

total = 200
max_nodes = 328
chunk_number = 1;

cplus_graph = "std::vector<std::vector<std::array<int, 2>>> GE;\n"
cplus_push_backs = ""
for i in range(0,total):
    num_nodes = random.randint(5, max_nodes)
    while True:
        G = nx.gnp_random_graph(num_nodes, np.random.rand())
        if G.number_of_edges() > 1:
            break 
        
    g_edges = str(G.number_of_edges())
    #"std::vector<std::array<int, 2>> E0 = "
    #"std::vector<std::vector<std::array<int, 2>>>"
    cplus_graph += "std::vector<std::array<int, 2>> E"+str(i)+" = { " 
    first = True
    for edge in G.edges:
        if first :
            cplus_graph += "{ "+ str(edge[0])+" , " + str(edge[1])+" }"
            first = False
        else:
            cplus_graph += ",{ "+ str(edge[0])+" , " + str(edge[1])+" }"
            
        
    cplus_graph += "}; \n" 
    cplus_push_backs += "GE.push_back(E"+str(i) +");\n" 
    

    # make_run_folder(f"ind{i}","ind.cc")
    # write_to_file(f"ind{i}/c_chunk.h",c_lines)
    # # Example usage
    # bash_command = "ind > graph.py"
    # run_bash_command(bash_command)
#cplus_graph = cplus_graph[:-2]+"}}\n;"

print(cplus_graph)
print( "void run_push_backs(){\n") ;
print(cplus_push_backs)
print("}")