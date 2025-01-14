#Import the necessary libraries 
import json
import cv2

import sqlite3

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import gensim
import gensim.corpora as corpora
from nltk.tokenize import word_tokenize

from prettytable import PrettyTable

#Establish connection to the database
connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()

#Load the features for the target features
with open('../database/total_target_features.json', 'r') as f:
    target_data = json.load(f)

# Load the category map features that are extracted for each label
with open('../database/category_map_kmeans.json', 'r') as f:
    kmeans_preprocess = json.load(f)

# Load the features for all the action centers
with open('../Outputs/Task_2/KMeans_latent.json', 'r') as f:
    cluster_centre = json.load(f)

target_labels = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow',
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch',
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']

Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
Dimensionality_Reduction_Map = {1: "PCA", 2: "SVD", 3: "LDA", 4:"KMeans"}

with open('../database/category_map.json', 'r') as f:
    category_map = json.load(f)

#Maps to store the feature values
kmeans_map = {}

#Calculate the Euclidean Distance
def euclidean(a, b):
    distance_res=0
    for i in range(0, len(a)):
        distance_res += (a[i] - b[i])**2
    return distance_res ** 0.5

def manhattan(a, b):
    res=0
    for i in range(0,len(a)):
        res = res + abs(a[i]-b[i])
    return res

#Function to calculate the mean
def feature_calculator(layer):
    feauter_np = np.array(layer)
    mean_feature = np.mean(feauter_np, axis=0)
    return mean_feature.tolist()

#Function to calculate the cluuster centres for all the video labels
def kmeans_preprocess_func(action, feature_space):
    layer_1=[]
    layer_2=[]
    layer_3=[]
    layer_1_cluster_centres = cluster_centre
    #For each category make a feature matrix 
    if feature_space==1:
        for videoname, category in category_map.items():
            if category==action:
                for video in target_data:
                    if videoname in video:
                        a=[]
                        a.extend(video[videoname])
                        layer_1.append(a[0])
        #Layer 1 Features for a action label
        layer_1_cluster_centres = cluster_centre
        layer_map=[]
        #For each video feature compute the distance with cluster center
        for i in range(len(layer_1)):
            a=[]
            for j in range(len(layer_1_cluster_centres)):
                dist=euclidean(layer_1[i],layer_1_cluster_centres[j])
                a.append(dist)
            layer_map.append(a)
        total_layer_5 = feature_calculator(layer_map)
    #For each category make a feature matrix 
    elif feature_space==2:
        for videoname, category in category_map.items():
            if category==action:
                for video in target_data:
                    if videoname in video:
                        a=[]
                        a.extend(video[videoname])
                        layer_2.append(a[1])
        layer_2_cluster_centres = cluster_centre
        layer_map=[]
        #For each video feature compute the distance with cluster center
        for i in range(len(layer_2)):
            a=[]
            for j in range(len(layer_2_cluster_centres)):
                dist=euclidean(layer_2[i],layer_2_cluster_centres[j])
                a.append(dist)
            layer_map.append(a)
        #For each category make a for layer2
        total_layer_5 = feature_calculator(layer_map)
    elif feature_space==3:
        for videoname, category in category_map.items():
            if category==action:
                for video in target_data:
                    if videoname in video:
                        a=[]
                        a.extend(video[videoname])
                        layer_3.append(a[2])
        #Similar find the features for the layer3
        layer_3_cluster_centres = cluster_centre
        layer_map=[]
        #For each video feature compute the distance with cluster center
        for i in range(len(layer_3)):
            a=[]
            for j in range(len(layer_3_cluster_centres)):
                dist=euclidean(layer_3[i],layer_3_cluster_centres[j])
                a.append(dist)
            layer_map.append(a)
        total_layer_5 = feature_calculator(layer_map)
    ####Integrate all the features for HoG
    elif feature_space==4:
        layer_4=[]
        #For each video extract the HoG features
        for videoname, category in category_map.items():
            if category==action:
                for video in target_data:
                    if videoname in video:
                        get_query_video_feature = f"SELECT {'BOF_HOG'} FROM data WHERE Video_Name = '{videoname}';"
                        c.execute(get_query_video_feature)
                        rows = c.fetchall()
                        cleaned_str = rows[0][0].strip("[]")
                        query_feature = list(map(int, cleaned_str.split()))
                        #query_feature = np.array(query_feature).reshape(1, -1)
                        layer_4.append(query_feature)
        #Make a numpy array for all the features which are HoG
        layer_4_cluster_centres = layer_1_cluster_centres
        layer_map=[]
        for i in range(len(layer_4)):
            a=[]
            for j in range(len(layer_4_cluster_centres)):
                dist=euclidean(layer_4[i],layer_4_cluster_centres[j])
                a.append(dist)
            layer_map.append(a)
        #Total features for layer4
        total_layer_5 = feature_calculator(layer_map)
    #Gather all the features for HoF
    else:
        layer_5=[]
        for videoname, category in category_map.items():
            if category==action:
                for video in target_data:
                    if videoname in video:
                        get_query_video_feature = f"SELECT {'BOF_HOF'} FROM data WHERE Video_Name = '{videoname}';"
                        c.execute(get_query_video_feature)
                        rows = c.fetchall()
                        cleaned_str = rows[0][0].strip("[]")
                        query_feature = list(map(int, cleaned_str.split()))
                        #query_feature = np.array(query_feature).reshape(1, -1)
                        layer_5.append(query_feature)
        #Make a numpy array for all the features which are HoF
        layer_5_cluster_centres = layer_1_cluster_centres
        layer_map=[]
        for i in range(len(layer_5)):
            a=[]
            for j in range(len(layer_5_cluster_centres)):
                dist=euclidean(layer_5[i],layer_5_cluster_centres[j])
                a.append(dist)
            layer_map.append(a)
        #Total features for HoG
        total_layer_5 = feature_calculator(layer_map)
    final_features =[total_layer_5]
    kmeans_map[action]=final_features
    return


#Function that lists m similar videos
def kmeans(label, feature_space,m):
    for video in target_labels:
        kmeans_preprocess_func(video, feature_space)
    with open('../database/category_map_kmeans.json', 'w') as f:
        json.dump(kmeans_map,f)
    #Load the category map features that are extracted for each label
    with open('../database/category_map_kmeans.json', 'r') as f:
        kmeans_preprocess = json.load(f)
    #Extract the label feature 
    query_features=kmeans_preprocess[label]
    query_feature = query_features[0]
    #For a specific cluster extract the centroids
    #cluster_centre = all_clusters[feature_space-1]
    
    res=[]
    #For the target videos gather the features
    for video in target_data:
        video_name =video.keys()
        layer_value_1=[]
        for key, value in video.items():
            #Extract the features for layer1, layer2 and layer3
            if feature_space==1 or feature_space==2 or feature_space==3:           
                video_name=key
                layer_values=value
                layer_value=layer_values[feature_space-1]
                a=[]
                #Calculate the Distance from the cluster centers
                for i in range(0, len(cluster_centre)):
                    dist = euclidean(layer_value, cluster_centre[i])
                    layer_value_1.append(dist)
            #Extract the features for HoG
            elif feature_space==4:
                video_name=key
                get_query_video_feature = f"SELECT {'BOF_HOG'} FROM data WHERE Video_Name = '{video_name}';"
                c.execute(get_query_video_feature)
                rows = c.fetchall()
                cleaned_str = rows[0][0].strip("[]")
                layer_value = list(map(int, cleaned_str.split()))
                for i in range(0, len(cluster_centre)):
                    dist = euclidean(layer_value, cluster_centre[i])
                    layer_value_1.append(dist)
            #Extract the features for HoF
            elif feature_space==5:
                video_name=key
                get_query_video_feature = f"SELECT {'BOF_HOF'} FROM data WHERE Video_Name = '{video_name}';"
                c.execute(get_query_video_feature)
                rows = c.fetchall()
                cleaned_str = rows[0][0].strip("[]")
                layer_value = list(map(int, cleaned_str.split()))
                layer_value_1=[]
                #Calculate the Distance from the cluster centers
                for i in range(0, len(cluster_centre)):
                    dist = euclidean(layer_value, cluster_centre[i])
                    layer_value_1.append(dist)
            #Calculate the distance between query video and current video
            distance = euclidean(query_feature, layer_value_1)
            res.append((distance, key))
    #Sort based on distance
    res.sort(key=lambda i:i[0])
    print(f"******The \"{m}\" most similar videos are: ********")
    #Printing the videos along with distance
    video_name = []
    for i in range(0, m):
        print(f'{res[i][1]} : {res[i][0]}')
        video_name.append(res[i][1])
    return

def get_Label_features(label, feature_space, latent_features, isLDA):

    label_query = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE Action_Label='{label}';"
    c.execute(label_query)
    rows = c.fetchall()

    cleaned_data = []
    if feature_space in [1, 2, 3]:
        for row in rows:
            cleaned_data.append(list(map(float, row[0].strip("[]").split(","))))
    elif feature_space in [4, 5]:
        for row in rows:
            cleaned_data.append(list(map(int, row[0].strip("[]").split())))

    max_len = max(len(lst) for lst in cleaned_data)
    padded_data = [lst + [0] * (max_len - len(lst)) for lst in cleaned_data]
    data = np.array(padded_data)

    averaged_array = np.mean(data, axis=0)
    reshaped_array = averaged_array.reshape(1, max_len)

    if isLDA:
        lda_model = gensim.models.LdaModel.load("../Outputs/Task_2/lda_model")
        dictionary = corpora.Dictionary.load("../Outputs/Task_2/dictionary")
        string_data = []
        for sublist in reshaped_array:
            string_data.append(' '.join(map(str, sublist)))
        return infer_new_document_lda(lda_model, dictionary, string_data)
    
    else:
        dim_reduced_array = np.dot(reshaped_array, latent_features)
        return dim_reduced_array

def compare_distances(label_data, feature_space, latent_features, m, isLDA):

    all_video_query = f"SELECT Video_Name, {Feature_Space_Map[feature_space]} FROM data WHERE videoID <= 2872;"
    c.execute(all_video_query)
    rows = c.fetchall()

    cleaned_data = []
    if feature_space in [1, 2, 3]:
        for row in rows:
            cleaned_data.append(list(map(float, row[1].strip("[]").split(","))))
    elif feature_space in [4, 5]:
        for row in rows:
            cleaned_data.append(list(map(int, row[1].strip("[]").split())))

    max_len = max(len(lst) for lst in cleaned_data)
    padded_data = [lst + [0] * (max_len - len(lst)) for lst in cleaned_data]
    data = np.array(padded_data)

    if isLDA:
        lda_model = gensim.models.LdaModel.load("../Outputs/Task_2/lda_model")
        dictionary = corpora.Dictionary.load("../Outputs/Task_2/dictionary")
        string_data = []
        for sublist in data:
            string_data.append(' '.join(map(str, sublist)))
        dim_reduced_data = infer_new_document_lda(lda_model, dictionary, string_data)
    else:
        dim_reduced_data = np.dot(data, latent_features)

    # Calculate the distance for all the video features to the query video.
    distances = cdist(label_data, dim_reduced_data, metric='euclidean').flatten()
    
    # Sort the items based on distances
    indices = np.argsort(distances)[:m]
    # Print the items
    print(f"\n {Feature_Space_Map[feature_space]} - Closest Videos")
    t = PrettyTable(["Rank", "Video Name", "Distance"])
    # Append the results to a list for visualisation
    rank = 1
    for idx in indices:
        t.add_row([rank, rows[idx][0], distances[idx]])
        rank += 1
    print(t)

    return dim_reduced_data

def infer_new_document_lda(lda_model, dictionary, new_document):
    
    # Preprocess the new document: tokenize
    new_tokens = [word_tokenize(string) for string in new_document]
    
    # Convert the new document to the bag-of-words representation using the trained dictionary
    new_bow = [dictionary.doc2bow(token) for token in new_tokens]
    
    # Get the topic distribution for the new document
    topic_distribution = [lda_model.get_document_topics(bow) for bow in new_bow]
    
    # Convert to a feature matrix (dimensionality-reduced representation)
    def topic_vector(lda_output, num_topics):
        vector = np.zeros(num_topics)
        for topic_num, prob in lda_output:
            vector[topic_num] = prob
        return vector
    
    feature_matrix = np.array([topic_vector(doc, lda_model.num_topics) for doc in topic_distribution])

    return feature_matrix


def main():

    label = input("Please provide the label: ")
    latent_Semantic = int(input("Please Provide the Latent Semantics 1 - PCA, 2 - SVD, 3 - LDA, 4 - KMeans: "))
    m = int(input("Please provide a value for m: "))
    feature_space = int(input("Select the Feature Space used in Task 2: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4 - HOG, 5 - HOF, 6 - Color Histogram : "))

    # Gather the feature space
    if latent_Semantic == 1:
        latent_features = np.load('../Outputs/Task_2/PCA_left_matrix.npy')
        label_data = get_Label_features(label, feature_space, latent_features, False)
        compare_distances(label_data, feature_space, latent_features, m, False)
    elif latent_Semantic == 2:
        latent_features = np.load('../Outputs/Task_2/SVD_right_matrix.npy')
        label_data = get_Label_features(label, feature_space, latent_features, False)
        compare_distances(label_data, feature_space, latent_features, m, False)
    elif latent_Semantic == 3:
        label_data = get_Label_features(label, feature_space, np.empty(1) , True)
        compare_distances(label_data, feature_space, np.empty(1), m, True)
    elif latent_Semantic == 4:
        kmeans(label, feature_space,m)
    else:
        print("Invalid input.")

main()
