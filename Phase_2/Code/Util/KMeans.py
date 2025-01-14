import json
import numpy as np
from sklearn.cluster import KMeans
import sqlite3
#Establish connection to database
connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()
with open('../database/total_target_features.json', 'r') as f:
    target_data = json.load(f)
with open('../database/category_map.json', 'r') as f:
    category_map = json.load(f)
with open('../database/videoID.json', 'r') as f:
    videoID = json.load(f)

kmeans_centroids = {}
kmeans_videoID_maps = {}

Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}

def euclidean(a, b):
    distance_res=0
    for i in range(0, len(a)):
        distance_res += (a[i] - b[i])**2
    return distance_res ** 0.5

#Define the kmeans clustering for the latent sematics "s" 
def kmeans_clustering(s, feature_model):
    total_features=[]
    #Extract the video name from the json
    for video in target_data:
        video_name =video.keys()
        for key, value in video.items():
            video_name=key
            layer_values=value
        #Check whether the video is even or odd
        if videoID[video_name]%2==0:
            layer=[]
            layer.append(videoID[video_name])
            #If feature is layer3 append the layer_3 features
            if feature_model==1:
                layer.append(layer_values[0])
            #If feature is layer4 append the layer_4 features
            elif feature_model==2:
                layer.append(layer_values[1])
            #If feature is avgpool append the avgpool features
            elif feature_model==3:
                layer.append(layer_values[2])
            elif feature_model==4:
                get_query_video_feature = f"SELECT {'BOF_HOG'} FROM data WHERE Video_Name = '{video_name}';"
                c.execute(get_query_video_feature)
                rows = c.fetchall()
                cleaned_str = rows[0][0].strip("[]")
                layer.append(list(map(int, cleaned_str.split())))
            elif feature_model==5:
                get_query_video_feature = f"SELECT {'BOF_HOF'} FROM data WHERE Video_Name = '{video_name}';"
                c.execute(get_query_video_feature)
                rows = c.fetchall()
                cleaned_str = rows[0][0].strip("[]")
                layer.append(list(map(int, cleaned_str.split())))
            total_features.append(layer)
    #Combine the entries with a video id and its values
    combined_data=[]
    for item in total_features:
        id=item[0]
        feature=item[1]
        combined_data.append([id]+feature)
    total_features=[]
    #Convert the matrix to the numpy array which includes both the videoID, features
    total_features = np.array(combined_data)
    #Before training using kmeans separate the videoID
    filtered_features = total_features[:,1:]
    kmeans  = KMeans(n_clusters=s, random_state=42)
    kmeans.fit(filtered_features)
    #Compute the Cluster Centres
    cluster_centre = kmeans.cluster_centers_
    cluster_centre_list = cluster_centre.tolist()
    video_weight={}
    for video in target_data:
        video_name =video.keys()
        for key, value in video.items():
            video_name=key
            layer_values=value
        #Check whether the video is even or odd
        if videoID[video_name]%2==0:
            layer=[]
            #If feature is layer3 append the layer_3 features
            if feature_model==1:
                layer=layer_values[0]
            #If feature is layer4 append the layer_4 features
            elif feature_model==2:
                layer=layer_values[1]
            #If feature is avgpool append the avgpool features
            elif feature_model==3:
                layer=layer_values[2]
            elif feature_model==4:
                get_query_video_feature = f"SELECT {'BOF_HOG'} FROM data WHERE Video_Name = '{video_name}';"
                c.execute(get_query_video_feature)
                rows = c.fetchall()
                cleaned_str = rows[0][0].strip("[]")
                layer = list(map(int, cleaned_str.split()))
            elif feature_model==5:
                get_query_video_feature = f"SELECT {'BOF_HOF'} FROM data WHERE Video_Name = '{video_name}';"
                c.execute(get_query_video_feature)
                rows = c.fetchall()
                cleaned_str = rows[0][0].strip("[]")
                layer = list(map(int, cleaned_str.split()))
            ans=[]
            for i in range(0, len(cluster_centre_list)):
                centre=cluster_centre_list[i]
                dist=euclidean(centre, layer)
                ans.append(dist)
            minimum_val=min(ans)
            video_weight[videoID[video_name]]=minimum_val
    sorted_video_weight = dict(sorted(video_weight.items(), key=lambda x:x[1], reverse=True))
    res=[sorted_video_weight]
    with open('../Outputs/Task_2/KMeans_latent.json', 'w') as f:
        json.dump(cluster_centre_list, f, indent=4)
    with open(f'../Outputs/Task_2/videoID-weight_KMeans_{Feature_Space_Map[feature_model]}.json', 'w') as f:
        json.dump(res, f, indent=4)
    print(f"******The \"{s}\" latent semantics are: ********")
    
    #Print the cluster centres
    for i in range(len(cluster_centre_list)):
        print(f"The cluster centre - {i+1} is: ", cluster_centre_list[i])
    return
