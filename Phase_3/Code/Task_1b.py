#Import all the necessary libraries
from Util.KMeanslatentmodel import latent_model_generator
from Util.KMeanslatentmodel import PCA
from Util.KMeanslatentmodel import SVD
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from collections import Counter
import mplcursors

#Select the videos from target labels
target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']

#Function to randomly initialise the centriods
def initialize_centroids(features, k):
    if features is None or features.shape[0] < k:
        raise ValueError("Error: Invalid `features` input or insufficient data points for k clusters.")
    #Depending on value k randomly select k features.
    indices = np.random.choice(features.shape[0], k, replace=False)
    return features[indices]

#For each of the centroids assign the features
def assign_clusters(features, centroids):
    distances = np.array([[np.linalg.norm(point - centroid) for centroid in centroids] for point in features])
    return np.argmin(distances, axis=1)

#Update the Centriods after recalcuation
def update_centroids(features, labels, k):
    return np.array([features[labels == i].mean(axis=0) for i in range(k)])

#Function to calculate K Clusters
def kmeans_clustering(features, k, max_iters=100, tol=0.1):
    #Initialize the centroids
    centroids = initialize_centroids(features, k)
    #Depeding on the tolerance calculate the centroids
    for _ in range(max_iters):
        labels = assign_clusters(features, centroids)
        new_centroids = update_centroids(features, labels, k)
        
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
    #return the labels and the centroids
    return labels, centroids

#Visualization using MDS Space
def visualize_clusters(features, labels, f_d, k):
    dimensions = [["PCA", "Layer3"], ["SVD", "AvgPool"], ["KMeans", "HOG"]]
    #Configure the MDS as 2D
    mds = MDS(n_components=2, random_state=0)
    #Transform the features
    features_2d = mds.fit_transform(features)
    
    #For each of data point assign a cluster
    plt.figure(figsize=(10, 8))
    for cluster in range(k):
        cluster_points = features_2d[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster+1}')
    plt.legend()
    plt.title("Cluster Visualization in 2D MDS Space")
    plt.xlabel(f"{dimensions[f_d-1][0]}")
    plt.ylabel(f"{dimensions[f_d-1][1]}")
    plt.show()
    
    print("Groups of Labels in Each Cluster:")
    for cluster in range(k):
        cluster_labels = np.where(labels == cluster)[0]
        print(f"Cluster {cluster+1}: Labels {cluster_labels}")

#Visualise as group of labels
def helper_visualise(features, labels, k, centroids, target_label_range, f_d):
    dimensions = [["PCA", "Layer3"], ["SVD", "AvgPool"], ["KMeans", "HOG"]]
    labels = assign_clusters(features, centroids)
    labels_list = labels.tolist()
    assignment = []
    #Gtaher the assignment for each lable
    for i in range(0, k):
        cluster = []
        for j in range(0, len(labels_list)):
            if labels_list[j]==i:
                cluster.append(j)
        assignment.append(cluster)
    label_assignment = []
    #For each assignment gather the action type or label
    for i in range(len(assignment)):
        cluster = assignment[i]
        curr_label = []
        for j in range(len(cluster)):
            element = cluster[j]
            for key, item in target_label_range.items():
                if item[0]<=element  and item[1]>=element:
                    curr_label.append(key)
        counter_label = Counter(curr_label)
        sorted_labels = [label for label, count in counter_label.most_common()]
        label_assignment.append(sorted_labels)
    
    #Visualise using matplotlib    
    for i in range(k):
        #plot the centroids
        cluster_points = features[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")   
    plt.scatter(centroids[:, 0], centroids[:, 1], label="Centroids", color="black", marker="x")
    plt.legend()
    #Import mpl coursors to add hovering
    cursor = mplcursors.cursor(hover=True)
    #Establish the connection to the cursor
    @cursor.connect("add")
    def on_add_annotation(sel):
        for i in range(k):
            cluster_points_2d = features[labels == i][:, :2]
            #Gather the cluster for the data point 
            if any((sel.target == cluster_points_2d).all(axis=1)):
                sel.annotation.set_text(f"Cluster {i}\nLabels: {', '.join(label_assignment[i][:8])}")
                break
    #Labeling using matplotlib
    plt.title("K-Means Group Clustering")
    plt.xlabel(f"{dimensions[f_d-1][0]}")
    plt.ylabel(f"{dimensions[f_d-1][1]}")
    plt.show()

#Main Function whether program executes
def main():
    #Choice of the latent model
    latent_model = int(input("Please provide the input for 1 - Layer3 + PCA, 2 - Avgpool + SVD, 3 - KMeans + HOG: "))
    #Provide the number of clusters
    k = int(input("Please provide the input for number of clusters: "))
    if latent_model == 1:
        #Gather the target label range and the latent model
        latent_model, target_label_range =  PCA(1)
        features = latent_model
        #kmeans clustering for the features
        labels, centroids = kmeans_clustering(features,k)
        #Visualise using MDS and Group of label clusters
        input_visualization = int(input("Please provide the visualization technique 1 - MDS, 2 - Group of Clusters: "))
        if input_visualization == 1:
            visualize_clusters(features, labels, 1, k=len(centroids))
        else:
            helper_visualise(features, labels, k, centroids, target_label_range, 1)
    elif latent_model ==2:
        #Gather the target label range and the latent model
        latent_model, target_label_range =  SVD(3)
        features = latent_model
        #kmeans clustering for the features
        labels, centroids = kmeans_clustering(features,k)
        #Visualise using MDS and Group of label clusters
        input_visualization = int(input("Please provide the visualization technique 1 - MDS, 2 - Group of Clusters: "))
        if input_visualization == 1:
            visualize_clusters(features, labels, 2,k=len(centroids))
        else:
            helper_visualise(features, labels, k, centroids, target_label_range, 2)
    else:
        #Gather the target label range and the latent model
        latent_model, target_label_range = latent_model_generator(4)
        features = np.array(latent_model)
        #kmeans clustering for the features
        labels, centroids = kmeans_clustering(features,k)
        #Visualise using MDS and Group of label clusters
        input_visualization = int(input("Please provide the visualization technique 1 - MDS, 2 - Group of Clusters: "))
        if input_visualization == 1:
            visualize_clusters(features, labels, 3,k=len(centroids))
        else:
            helper_visualise(features, labels, k, centroids, target_label_range, 3)
main()
