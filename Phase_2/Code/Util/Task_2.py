import os
import json

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


# Dictionaries to map (tau, sigma) values to Cluster Representatives
HOG_centers = {}
HOF_centers = {}

stip_column_labels = ["point-type", "x", "y", "t", "sigma2", "tau2", "detector-confidence"]
for i in range(1, 73):
    stip_column_labels.append("dscr-hog (" + str(i) + ")")
for i in range(1, 91):
    stip_column_labels.append("dscr-hof (" + str(i) + ")")

tau_values = [2, 4]
sigma_values = [4, 8, 16, 32, 64, 128]


def create_cluster_representatives():
    non_target_path = '../dataset_stips/non_target_videos'

    # Array to store 400 highest confidence STIPs for each non-target video
    non_target_data = []

    for non_target_videos in os.listdir(non_target_path):
        video_path = os.path.join(non_target_path, non_target_videos)

        try:
            temp_df = pd.read_csv(video_path, sep="\t", comment='#', header=None)

            # select 400 highest samples sorted by column index 6 (detector-confidence)
            temp_df = temp_df.nlargest(400, 6)

            non_target_data.append(temp_df)
        
        except pd.errors.EmptyDataError:
            print("no data in ", non_target_videos)
            continue
    
    # Convert Array to Pandas DataFrame
    non_target_df = pd.concat(non_target_data, ignore_index=True)

    # Drops the last column as it is an empty column (due to how STIP data provided is formatted)
    non_target_df = non_target_df.dropna(axis=1)
    
    non_target_df.columns = stip_column_labels


    for t in tau_values:
        for s in sigma_values:
            # Sample 10000 STIPs for each pair of tau2, sigma2 pairs
            filtered_df = non_target_df[(non_target_df["tau2"] == t) & (non_target_df["sigma2"] == s)].sample(10000)

            # Take the HoG and HoF features
            X_HOG = filtered_df.loc[:, "dscr-hog (1)":"dscr-hog (72)"]
            X_HOF = filtered_df.loc[:, "dscr-hof (1)":"dscr-hof (90)"]

            kmeans = KMeans(n_clusters=40)

            kmeans.fit(X_HOG)
            HOG_centers[str((t, s))] = kmeans.cluster_centers_.tolist()

            kmeans.fit(X_HOF)
            HOF_centers[str((t, s))] = kmeans.cluster_centers_.tolist()

    # Export the Cluster Representatives
    with open("Util/HoG_cluster_representatives.json", "w") as fp:
        json.dump(HOG_centers, fp)
    with open("Util/HoF_cluster_representatives.json", "w") as fp:
        json.dump(HOF_centers, fp)


def find_closest_clusters(x, y):
    distances = euclidean_distances(x, y)
    return np.argmin(distances, axis=1)


def get_HoG_HoF_features(target_path):
    try:
        df = pd.read_csv(target_path, sep="\t", comment='#', header=None)
        df = df.nlargest(400, 6)
        df = df.dropna(axis=1)
        df.columns = stip_column_labels
    
    except pd.errors.EmptyDataError:
        print("no data in ", target_path)
        return [], []

    hog_histograms = []
    hof_histograms = []

    for t in tau_values:
        for s in sigma_values:

            filtered_df = df[(df["tau2"] == t) & (df["sigma2"] == s)]
            if filtered_df.shape[0] != 0:
                hog_cluster_Id = find_closest_clusters(filtered_df.loc[:, "dscr-hog (1)":"dscr-hog (72)"], HOG_centers[str((t, s))])
                hof_cluster_Id = find_closest_clusters(filtered_df.loc[:, "dscr-hof (1)":"dscr-hof (90)"], HOF_centers[str((t, s))])

                hog_histogram, bin_edges = np.histogram(hog_cluster_Id, bins=np.arange(41))
                hof_histogram, bin_edges = np.histogram(hof_cluster_Id, bins=np.arange(41))

            elif filtered_df.shape[0] == 0:
                    hog_histograms.append(np.zeros(480))
                    hof_histograms.append(np.zeros(480))

            hog_histograms.append(hog_histogram)
            hof_histograms.append(hof_histogram)
    
    bof_HOG_descriptor = np.concatenate(hog_histograms)
    bof_HOF_descriptor = np.concatenate(hof_histograms)

    return bof_HOG_descriptor, bof_HOF_descriptor
