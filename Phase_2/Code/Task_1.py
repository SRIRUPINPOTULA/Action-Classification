#Import all the libraries
import json
import cv2
import sqlite3
import numpy as np
#Establish a connection with Database
connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()

#Read the HoG,HoF feature to label mapping
with open('../database/HoG_HoF_feature_label_representation.json', 'r') as f:
    HoG_features_extracted = json.load(f)

#HoG function that lists the m similar videos
def HoG(query_video, layer_number, l):
    #Feature for the query video
    get_query_video_feature = f"SELECT {'BOF_HOG'} FROM data WHERE Video_Name = '{query_video}';"
    c.execute(get_query_video_feature)
    rows = c.fetchall()
    cleaned_str = rows[0][0].strip("[]")
    query_feature = list(map(int, cleaned_str.split()))
    res=[]
    # For all the labels compute the distance using cosine similarity
    for key, value in HoG_features_extracted.items():
        distance = cosine_similarity(query_feature, value[layer_number-1])
        res.append((distance, key))
    #Sort inorder to get the m similar videos
    res.sort(key=lambda i:i[0], reverse=True)
    print(f"******The \"{l}\" most similar labels for the video are: ********")
    for i in range(0, l):
        print(f'{res[i][1]} : {res[i][0]}') 

#HoG function that lists the m similar videos using HoF features        
def HoF(query_video, layer_number, l):
    #HoG function that lists the m similar videos
    get_query_video_feature = f"SELECT {'BOF_HOF'} FROM data WHERE Video_Name = '{query_video}';"
    c.execute(get_query_video_feature)
    rows = c.fetchall()
    cleaned_str = rows[0][0].strip("[]")
    query_feature = list(map(int, cleaned_str.split()))
    #query_feature = np.array(query_feature).reshape(1, -1)
    res=[]
    # For all the labels compute the distance using cosine similarity
    for key, value in HoG_features_extracted.items():
        distance = cosine_similarity(query_feature, value[layer_number-1])
        res.append((distance, key))
    #Sort inorder to get the m similar videos
    res.sort(key=lambda i:i[0], reverse=True)
    print(f"******The \"{l}\" most similar labels for the video are: ********")
    for i in range(0, l):
        print(f'{res[i][1]} : {res[i][0]}')  
 
#Cosine Similarity Function
def cosine_similarity(a, b):
    dot_sum=0
    list_a=0
    list_b=0
    #Compute the Magnitude of list a
    for i in range(0,len(a)):
        list_a += a[i]**2
    list_a=list_a ** 0.5
    #Compute the Magnitude of list b
    for i in range(0,len(a)):
        list_b += b[i]**2
    list_b = list_b ** 0.5
    #Compute the dot product of list a to list b elements
    for i in range(0,len(a)):
        dot_sum += a[i] * b[i]
    final_ans= dot_sum/(list_a * list_b)
    return final_ans

#Nearest Search function is used to find the m similar labels for layer3, layer4, avgpool layers
def nearest_search(query_video, layer_number, l):
    #Read the json files
    with open('../database/total_target_features.json', 'r') as f:
        target_data = json.load(f)
    with open('../database/total_non_target_features.json', 'r') as f:
        non_target_data = json.load(f)
    with open('../database/feature_label_representation.json', 'r') as f:
        features_extracted = json.load(f)
    layer=[]
    found=False
    #Find the video is in target or non target videos
    for video in target_data:
        if query_video in video:
            layer.extend(video[query_video])
            found=True
            break
    if found ==False:
        for video in non_target_data:
            if query_video in video:
                layer.extend(video[query_video])
                found=True
                break
    #gather the feature vector
    if layer_number==1:
        layer = layer[0]
    elif layer_number==2:
        layer = layer[1]
    else:
        layer=layer[2]   
    res=[]
    #For the query video compute the distance with available feature label
    for key, value in features_extracted.items():
        distance = cosine_similarity(layer, value[layer_number-1])
        res.append((distance, key))
    #Sort the list
    res.sort(key=lambda i:i[0], reverse=True)
    print("******The \"m\" most similar labels for the video are: ********")
    for i in range(0, l):
        print(f'{res[i][1]} : {res[i][0]}')
#main function that takes the input from the user
def main():
    #Gather the video name or id.
    input_type = int(input("Provide the 1 - Video File Name or 2 - VideoID: "))
    if input_type==1:
        video_name = input("Provide the  Video File Name: ")
    else:
        video_number = int(input("Provide the  VideoID: "))
        with open('../database/videoID.json', 'r') as f:
            videoID = json.load(f)
        for key, value in videoID.items():
            if value==video_number:
                video_name=key
                break  
    #Gather the feature space
    feature_space = int(input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF, 6 - Color Histogram : "))
    m = int(input("Provide the value of m: "))
    if feature_space==1 or feature_space==2 or feature_space==3:
        nearest_search(video_name, feature_space, m)
    elif feature_space==4:
        videos = HoG(video_name, 1,m)
    elif feature_space==5:
        videos = HoF(video_name, 2,m)
    else:
        print("Histograms")
main()

        
