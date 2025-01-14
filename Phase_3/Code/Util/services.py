import numpy as np

import sqlite3

from Util.PCA import PCA
from Util.SVD import SVD
from Util.KMeans import kMeans

class ServiceClass():

    # Establish connection to the database
    connection = sqlite3.connect('../Database/Phase_3.db')
    c = connection.cursor()

    target_labels = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                    'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                    'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                    'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']
    
    all_feature_space_map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    feature_space_map = {1: "Layer_3", 2: "AvgPool", 3: "BOF_HOG"}
    dimensionality_reduction_map = {1: "PCA", 2: "SVD", 3: "KMeans"}

    # Clean the extracted values and make a list of features
    def clean_db_data(rows, latent_space):
        cleaned_data = []
        video_names = []
        if latent_space in [1, 2]:
            for row in rows:
                cleaned_data.append(list(map(float, row[1].strip("[]").split(","))))
                video_names.append(row[0])
        elif latent_space == 3:
            for row in rows:
                cleaned_data.append(list(map(int, row[1].strip("[]").split())))
                video_names.append(row[0])

        max_len = max(len(lst) for lst in cleaned_data)
        padded_data = [lst + [0] * (max_len - len(lst)) for lst in cleaned_data]
        fs_data = np.array(padded_data)
        return fs_data, video_names

    def get_reduced_dimensions(fs_data, latent_space, s):
        if (latent_space == 1):
            dim_reduced_array = PCA(fs_data, s)
        elif (latent_space == 2) :
            dim_reduced_array = SVD(fs_data, s)
        else:
            # Kmeans does not allow more clusters than the number of samples
            if (s > len(fs_data)):
                s = len(fs_data)
            dim_reduced_array = kMeans(fs_data, s)

        return dim_reduced_array
    
    def get_data_from_db(query, latent_space, s):
        ServiceClass.c.execute(query)
        rows = ServiceClass.c.fetchall()

        fs_data, video_names = ServiceClass.clean_db_data(rows, latent_space)

        data = ServiceClass.get_reduced_dimensions(fs_data, latent_space, s)

        return data, video_names
