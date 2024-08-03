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
from time_elapsed import make_time_stamp


# Graph helper functions.
def neighborhood(G, v):
    return list(nx.neighbors(G, v))

def set_neighborhood(G, nodes):
    N = set()
    for n in nodes:
        N |= set(neighborhood(G, n))
    return list(N)

def closed_neighborhood(G, v):
    return list(set(neighborhood(G, v)).union([v]))

# Set up a linear-integer optimization formulation to compute exactly a largest
# independent set in a graph.
def max_independent_set_ilp(G):
    prob = LpProblem("max-independent-set", LpMaximize)
    variables = {
        node: LpVariable("x{}".format(i + 1), 0, 1, LpBinary)
        for i, node in enumerate(G.nodes())
    }
    prob += lpSum(variables)
    for e in G.edges():
        prob += variables[e[0]] + variables[e[1]] <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    solution_set = {node for node in variables if variables[node].value() == 1}
    return solution_set

# Compute the independence number of a graph exactly.
def independence_number(G):
    return len(max_independent_set_ilp(G))

# Function to convert adjacency matrix and heatmap to fixed-size image.
def convert_to_heatmap_image(G, target_size=64):
    adj_matrix = nx.to_numpy_array(G)
    matrix_size = adj_matrix.shape[0]

    if matrix_size < target_size:
        padded_matrix = np.zeros((target_size, target_size), dtype=int)
        padded_matrix[:matrix_size, :matrix_size] = adj_matrix
    elif matrix_size > target_size:
        image = Image.fromarray(adj_matrix)
        padded_matrix = np.array(image.resize((target_size, target_size), Image.BILINEAR))
    else:
        padded_matrix = adj_matrix

    degree_dict = dict(G.degree())
    degrees = np.array([degree_dict[node] for node in G.nodes()])
    heatmap = np.zeros((target_size, target_size), dtype=float)

    for i, node in enumerate(G.nodes()):
        if i < target_size:
            heatmap[i, i] = degrees[i]

    combined_matrix = padded_matrix + heatmap
    combined_image = (combined_matrix / combined_matrix.max() * 255).astype(np.uint8)

    return Image.fromarray(combined_image, 'L')

model = models.load_model("./independence_model/independence_cnn.keras")
time_elapsed = make_time_stamp()


# Big Test

# Generate and process multiple random graphs
num_graphs = 30
num_nodes = 30
print(f"Going for big test. Random graphs {num_graphs} with {num_nodes}")
results = []

for i in range(num_graphs):
    #G = nx.gnp_random_graph(n, np.random.rand()) -- 'n' was defined much earlier
    G = nx.gnp_random_graph(num_nodes, np.random.rand())

    # Convert adjacency matrix to image with heatmap
    adj_image = convert_to_heatmap_image(G)

    # Prepare the image for the model
    input_image = np.array(adj_image).reshape(1, 64, 64, 1)  # Add batch dimension and channel dimension

    # Predict the independence number using the trained CNN
    predicted_independence_number = model.predict(input_image, verbose="0")[0][0]
    

    # Compute the actual independence number
    actual_independence_number = independence_number(G)

    # Store the results
    results.append({
        "Graph Index": i,
        "Predicted Independence Number": predicted_independence_number,
        "Actual Independence Number": actual_independence_number
    })

print(results)
# Create a DataFrame to hold the results
results_df = pd.DataFrame(results)

# Display the results as a table
print(results_df)

# Plot the results as a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(results_df["Graph Index"], results_df["Predicted Independence Number"], label="Predicted", color='blue', alpha=0.6)
plt.scatter(results_df["Graph Index"], results_df["Actual Independence Number"], label="Actual", color='red', alpha=0.6)
plt.plot(results_df["Graph Index"], results_df["Actual Independence Number"], color='red', linestyle='dotted')
plt.plot(results_df["Graph Index"], results_df["Predicted Independence Number"], color='blue', linestyle='dotted')
plt.xlabel("Graph Index")
plt.ylabel("Stability Number")
plt.title("Predicted vs Actual Stability Numbers Random Graphs")
plt.ylim(0, num_nodes+1)
plt.legend()
#plt.show()

# Optionally, save the figure
plt.savefig("stability_number_prediction_comparison.png")

