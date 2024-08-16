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
    print("Finished a graph heat map")

    return Image.fromarray(combined_image, 'L')

# Function to generate random graphs and calculate their independence number.
def generate_independence_number_data(num_graphs, max_nodes):
    graphs = []
    independence_numbers = []
    total = num_graphs;
    print(f"Making independence for {total} graphs")
    
    for _ in range(num_graphs):
        _ = time_elapsed()
        num_nodes = random.randint(5, max_nodes)
        G = nx.gnp_random_graph(num_nodes, np.random.rand())
        ind_num = independence_number(G)
        graphs.append(G)
        independence_numbers.append(ind_num)
        total -= 1
        time = time_elapsed()
        print("Graphs left ",total," time take for last one",time)
    return graphs, independence_numbers

time_elapsed = make_time_stamp()


# print("Starting with nx random graphs 45,0.5")
# G = nx.gnp_random_graph(45, 0.5)
# nx.draw(G)
# #plt.show()
# plt.savefig("./test_random_graph.png")
# print("Saved test image")
# print("Starting convert to to heat map")
# _ = time_elapsed()
# image = convert_to_heatmap_image(G)
# plt.imshow(image, cmap='viridis', aspect='auto')
# #plt.show()
# plt.savefig("./test_random_graph_heat_map.png")

# # Create graphs with different numbers of vertices
# graphs = [
#     nx.gnp_random_graph(10, 0.5),
#     nx.gnp_random_graph(25, 0.5),
#     nx.gnp_random_graph(45, 0.5),
#     nx.gnp_random_graph(65, 0.5)
# ]

# # Create a figure with 2x2 subplots
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# # Titles for the subplots
# titles = ['10 vertices', '25 vertices', '45 vertices', '65 vertices']

# for i, (G, ax) in enumerate(zip(graphs, axs.flatten())):
#     image = convert_to_heatmap_image(G)
#     ax.imshow(image, cmap='viridis', aspect='auto')
#     ax.set_title(titles[i])
#     ax.axis('off')  # Hide the axes

# plt.tight_layout()
# #plt.show()
# plt.savefig("./TwoXTwo_Subplots.png")


# Generate dataset of 2,000 random graphs with varying node sizes 10 <= n <= 64.
num_graphs = 2_000
max_nodes = 64# 64
print(f" Creating dataset. {num_graphs} random graphs with maximumm nodes {max_nodes}.\n To make graphs with their independence number, \n two lists one for graphs and one for independence numbers ")

_ = time_elapsed()
graphs, independence_numbers = generate_independence_number_data(num_graphs, max_nodes)

# Convert graphs to images.
target_size = 64 #327# 64
print(f"Converting {target_size} graphs to heat map images.")
_ = time_elapsed()
images = [convert_to_heatmap_image(graph, target_size) for graph in graphs]
X = np.array([np.array(image).reshape(target_size, target_size, 1) for image in images])
y = np.array(independence_numbers)

# Split dataset into training and testing.
split_index = int(0.8 * num_graphs)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

time_taken = time_elapsed()
print(f'Finished making graphs in {time_elapsed} seconds.')

# Define the CNN model.
def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))  # Output layer for regression

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

# Instantiate and train the model.
print(f"Going for model creation and training the model")
# reset time start by calling time elapsed since last call
_ = time_elapsed()
input_shape = (target_size, target_size, 1)
model = create_model(input_shape)
model.summary()
history = model.fit(X_train, y_train, epochs=15, validation_split=0.2, verbose="2")
print(f"Model trained in {time_elapsed()} seconds.")

# Generate a random graph with n nodes.
n = 30
print(f"Going for small test of trained CNN model with {n} random graphs. ")
G = nx.gnp_random_graph(n, np.random.rand())

# Convert adjacency matrix to image with heatmap.
adj_image = convert_to_heatmap_image(G)

# Prepare the image for the model.
input_image = np.array(adj_image).reshape(1, 64, 64, 1)  # Add batch dimension and channel dimension.

# Predict the independence number using the trained CNN.
_ = time_elapsed()
predicted_independence_number = model.predict(input_image,verbose="0")
print(f"Predicted independence number for the {n}-node graph: {predicted_independence_number[0][0]} in {time_elapsed()} seconds")

# Compute the actual independence number using linear-integer programming for comparison.
actual_independence_number = independence_number(G)
print(f"Computed independence number for the {n}-node graph: {actual_independence_number} in {time_elapsed()} seconds")

model.save("./independence_model/independence_cnn_327.keras")
"""
# Big Test

# Generate and process multiple random graphs
num_graphs = 30
num_nodes = 30
print(f"Going for big test. Random graphs {num_graphs} with {num_nodes}")
results = []
_ = time_elapsed()
time_to_predict = 0

for _ in range(num_graphs):
    #G = nx.gnp_random_graph(n, np.random.rand()) -- 'n' was defined much earlier
    G = nx.gnp_random_graph(num_nodes, np.random.rand())

    # Convert adjacency matrix to image with heatmap
    adj_image = convert_to_heatmap_image(G)

    # Prepare the image for the model
    input_image = np.array(adj_image).reshape(1, 64, 64, 1)  # Add batch dimension and channel dimension

    # Predict the independence number using the trained CNN
    _ = time_elapsed()
    predicted_independence_number = model.predict(input_image, verbose="0")[0][0]
    time_to_predict = time_elapsed()

    # Compute the actual independence number
    actual_independence_number = independence_number(G)

    # Store the results
    results.append({
        "Graph Index": _,
        "Predicted Independence Number": predicted_independence_number,
        "Actual Independence Number": actual_independence_number,
        "Time to predict": time_to_predict
    })

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

"""