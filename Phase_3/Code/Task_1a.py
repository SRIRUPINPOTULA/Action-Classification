# Import the necessary libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import json

from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Util.services import ServiceClass as Service

# Get Latent Space for each label
def get_latent_space(latent_space):
    data_map = {}
    video_names_map = {}
    
    # Get the inherent dimensionality for each label from Task 0a
    with open(f'../Outputs/inherent_dim_map_{Service.feature_space_map[latent_space]}.json', 'r') as file:
        inherent_dim_map = json.load(file)

    for label in Service.target_labels:

        s = inherent_dim_map[label]

        label_query = f"SELECT Video_Name, {Service.feature_space_map[latent_space]} FROM data WHERE videoID % 2 == 0 AND Action_Label='{label}';"

        data, video_names = Service.get_data_from_db(label_query, latent_space, s)

        data_map[label] = data
        video_names_map[label] = video_names
    
    return data_map, video_names_map

# Get an initial default trheshold value for each Latent Space
def get_default_threshold(latent_space):
    if latent_space == 1:
        return 1
    elif latent_space == 2:
        return 10
    elif latent_space == 3:
        return 50
    else:
        return 1

# Change threshold value based on the ratio of clusters in graph to clusters needed by user
def get_threshold(threshold, ratio = None):
    if ratio == None:
        return threshold
    elif ratio > 1:
        threshold = threshold * 1.5
    elif ratio < 0.5:
        threshold = threshold * 0.8
    
    return threshold

def create_adjacency_matrix(data, threshold):
    num_videos = len(data)
    adjacency_matrix = np.zeros((num_videos, num_videos), dtype=int)

    for i in range(num_videos):
        for j in range(num_videos):
            # Ignore diagonal of matrix
            if i != j:
                distance = np.linalg.norm(data[i] - data[j])
                if distance <= threshold:
                    adjacency_matrix[i][j] = 1
    return adjacency_matrix

# Find all the connected components in the graph
def find_connected_components(adj_matrix):
    n = len(adj_matrix)
    visited = [False] * n
    components = []

    # Depth first search to traverse graph
    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, component)

    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)

    return components

# Count the number of connected components
# based on the index of second-smallest eigen value of laplacian matrix
def get_conn_comp_count(eigenvalues):
    for index in range(len(eigenvalues)):
        # Eigen value is being computed as a very small number instead of 0
        if eigenvalues[index] < 1e-5:
            continue
        else:
            return index

def partition(adj_matrix, nodes):
        
        # Degree matrix: Diagonal matrix of node degrees
        degree_matrix = np.diag(adj_matrix.sum(axis=1))

        # Laplacian matrix: D - A
        laplacian = degree_matrix - adj_matrix

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        # Use the Fiedler vector (second smallest eigenvector) for partitioning
        index = get_conn_comp_count(eigenvalues)
        fiedler_vector = eigenvectors[:, index]

        # Split nodes based on the sign of the Fiedler vector
        positive_set = [nodes[i] for i in range(len(nodes)) if fiedler_vector[i] > 0]
        negative_set = [nodes[i] for i in range(len(nodes)) if fiedler_vector[i] <= 0]

        return positive_set, negative_set

def get_sign_clusters(data, num_clusters, latent_space):
    # Get an initial default threshold value
    default_threshold = get_default_threshold(latent_space)
    threshold = get_threshold(default_threshold)
    adjacency_matrix = create_adjacency_matrix(data, threshold)

    # find all the connected components
    clusters = find_connected_components(adjacency_matrix)
    # print("Initial cluster number: ", len(clusters), "\nInitial clusters: \n", clusters)

    # Compute ratio to modify threshold value
    ratio = len(clusters) / num_clusters
    # print(f"ratio: {ratio}, threshold: {threshold}, num_clusters: {len(clusters)}, clusters_expected: {num_clusters}")

    while ratio > 1 or ratio < 0.5:
        threshold = get_threshold(threshold, ratio)
        adjacency_matrix = create_adjacency_matrix(data, threshold)
        clusters = find_connected_components(adjacency_matrix)
        ratio = len(clusters) / num_clusters
        # print(f"ratio: {ratio}, threshold: {threshold}, num_clusters: {len(clusters)}, clusters_expected: {num_clusters}")

    while len(clusters) < num_clusters:
        # Find Largest Cluster based on CLuster Distance metric
        largest_cluster_idx = np.argmax([len(c) for c in clusters])
        largest_cluster = clusters.pop(largest_cluster_idx)

        # Get the subgraph adjacency matrix for this cluster
        subgraph_adj_matrix = adjacency_matrix[np.ix_(largest_cluster, largest_cluster)]

        # Partition the largest cluster
        positive_set, negative_set = partition(subgraph_adj_matrix, largest_cluster)

        # Add the new clusters to the list
        clusters.append(positive_set)
        clusters.append(negative_set)
    
    return clusters

# Visualization using MDS Space
def visualize_MDS(data, cluster):
    # Latent Space 1 and 2 contain complex numbers.
    real_data = np.hstack((data.real, data.imag))
    distance_matrix = pairwise_distances(real_data, metric='euclidean')

    # Flatten clusters into a list of indices and assign cluster labels
    cluster_labels = np.zeros(data.shape[0], dtype=int)
    for cluster_idx, indices in enumerate(cluster):
        cluster_labels[indices] = cluster_idx

    # Apply MDS to reduce the data to 2D
    mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
    data_mds = mds.fit_transform(distance_matrix)

    # Visualize the data using clusters
    plt.figure(figsize=(10, 8))

    # Use colormap to represent upto 20 clusters
    colors = plt.colormaps['tab20'](np.linspace(0, 1, len(cluster)))

    for cluster_idx, color in enumerate(colors):
        cluster_points = data_mds[cluster_labels == cluster_idx]
        plt.scatter(cluster_points[:, 0],
                    cluster_points[:, 1], 
                    label=f'Cluster {cluster_idx+1}', 
                    color=color)

    plt.legend()
    plt.title("Cluster Visualization in 2D MDS Space")
    plt.grid(True)
    plt.show()

# Visualization using Video Thumbnails
def visualize_thumbnails(video_names, cluster):
    image_path = "../Database/Thumbnails_Names/"
    image_format = ".jpg"
    image_lists = [[image_path + video_names[i] + image_format for i in sublist] for sublist in cluster]

    # Loop through each sublist and create a separate plot for each
    for index, row in enumerate(image_lists):
        # Create a new figure for each row
        plt.figure(figsize=(15, 5))
        figure_title = f"Cluster {index+1}"
        plt.suptitle(figure_title, fontsize=16, fontweight='bold', color='black')
        
        # Loop through each image in the current row
        for j, image_path in enumerate(row):
            img = mpimg.imread(image_path)
            
            # Create a subplot for each image in the row
            ax = plt.subplot(1, len(row), j + 1)
            ax.imshow(img)
            ax.axis('off')

            label = (image_path.split('/')[-1])[:-4]
            ax.text(0.5, 0.05, label, ha='center', va='top', transform=ax.transAxes, color='white', fontsize=8, fontweight='bold')
        
        # Display the plot for the current row
        plt.tight_layout()
        plt.show(block=False)

# Visualize the clusters until User exits
def visualize(data, clusters, video_names):
    while True:
        vis_method = int(input("\nHow would you like to visualize the Clusters: 1- MDS space or 2- Group of Video Thumbnails: "))

        if vis_method not in [1 , 2]:
            print("Invalid input. Please choose either '1' or '2'.")
            continue

        while True:
            print(f"Target Labels:\n{Service.target_labels}")
            label = input(f"Choose a label. Enter 0 to stop: ").strip().lower()
            
            if label == '0':
                break
            elif label in Service.target_labels:
                print(f"{label}: {clusters[label]}")
                if vis_method == 1:
                    visualize_MDS(data[label], clusters[label])
                else:
                    visualize_thumbnails(video_names[label], clusters[label])
            else:
                print("Invalid choice")
        
        next_action = input("Would you like to switch visualization methods or exit? (switch/exit): ").strip().lower()
        
        if next_action == 'exit':
            print("Task complete")
            break
        elif next_action != 'switch':
            print("Invalid input. Please choose 'switch' or 'exit'.")
            continue

def main():
    latent_space = int(input("Select a latent space: 1 - layer3 + PCA, 2 - avgpool + SVD, 3 - HOG + KMeans: "))
    sig_clusters = int(input("Select the number of c most significant clusters to be selected: "))

    data, video_names = get_latent_space(latent_space)

    clusters = {}
    for label in data:
        clusters[label] = get_sign_clusters(data[label], sig_clusters, latent_space)

    visualize(data, clusters, video_names)


if __name__ == "__main__":
    main()
