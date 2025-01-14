# Import libraries
import json
import cv2

import numpy as np
from scipy.spatial.distance import cdist
from prettytable import PrettyTable

import sqlite3

from Util.Visualize import Visualize_HoG_HoF


#Importing features from json files
with open('../database/videoID.json', 'r') as f:
    videoID = json.load(f)
with open('../database/total_target_features.json', 'r') as f:
    target_data = json.load(f)
with open('../database/total_non_target_features.json', 'r') as f:
    non_target_data = json.load(f)
# with open('../database/feature_label_representation.json', 'r') as f:
#     features_extracted = json.load(f)

#Establish connection to database
connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()

#Visualisation function
def visualise(video_file):
    #Captured all the video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Couldnot read the video")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Visualizing the video frame using OpenCV
        cv2.imshow(f"The captured frame is: {video_file}", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Release and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

#Function to define Euclidean Distance
def euclidean(a, b):
    distance_res=0
    for i in range(0, len(a)):
        distance_res += (a[i] - b[i])**2
    return distance_res ** 0.5
# Function to find the m most similar video
def BOF(query_video, feature, l):
    #Gather the video features for the query video from the database
    get_query_video_feature = f"SELECT {feature} FROM data WHERE Video_Name = '{query_video}';"
    c.execute(get_query_video_feature)
    rows = c.fetchall()

    cleaned_str = rows[0][0].strip("[]")
    query_feature = list(map(int, cleaned_str.split()))
    query_feature = np.array(query_feature).reshape(1, -1)
    #Gather the video features that are from target video and have even id.
    get_all_video_feature = f"SELECT Video_Name, {feature} FROM data WHERE videoID % 2 = 0 AND Action_Label IS NOT NULL;"
    c.execute(get_all_video_feature)
    rows = c.fetchall()

    all_feature = []
    for row in rows:
        all_feature.append(list(map(int, row[1].strip("[]").split())))
    # Calculate the distance for all the video features to the query video.
    distances = cdist(query_feature, all_feature, metric='euclidean').flatten()
    # Sort the items based on distances
    indices = np.argsort(distances)[:l]
    # Print the items
    print(f"\n {feature} - Closest Videos to, {query_video}")
    t = PrettyTable(["Rank", "Video Name", "Distance"])
    # Append the results to a list for visualisation
    rank = 1
    result_names = []
    for idx in indices:
        t.add_row([rank, rows[idx][0], distances[idx]])
        result_names.append(rows[idx][0])
        rank += 1
    print(t)
    return result_names

#Function to list the m similar videos 
def layer3_implementation(query_video, layer_number, l):
    found = False
    with open('../database/total_target_features.json', 'r') as f:
        target_data = json.load(f)
    with open('../database/total_non_target_features.json', 'r') as f:
        non_target_data = json.load(f)
        
    layer=[]
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
    if layer_number==1:
        layer = layer[0]
    elif layer_number==2:
        layer = layer[1]
    else:
        layer=layer[2]
    res = []
    #Extract the layer and compare it with all other videos
    for i in target_data:
        for key, value in i.items():
            video_name=key
            if videoID[video_name]%2==0:
                curr_feature = value[layer_number-1]
                distance = euclidean(layer, curr_feature)
                res.append((distance, key))
    #Sort the Result based on Distances
    res.sort(key=lambda i:i[0])
    print(f"******The \"{l}\" most similar videos are: ********")
    video_name = []
    #Print the videos
    for i in range(0, l):
        print(f'{res[i][1]} : {res[i][0]}')
        video_name.append(res[i][1])
    return video_name
        
# Main function that prompts the user to initiate the code.
def main():
    # The user is prompted for input of video name
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
    #Gather the feature space choice from the user  
    feature_space = int(input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF, 6 - Color Histogram : "))
    m = int(input("Provide the value of m: "))
    # Conditions to call the functions
    if feature_space==1 or feature_space==2 or feature_space==3:
        videos=layer3_implementation(video_name, feature_space, m)
    elif feature_space==4:
        videos=BOF(video_name, 'BOF_HOG', m)
    elif feature_space==5:
        videos=BOF(video_name,'BOF_HOF', m)
    else:
        print("Histograms")
    #Input choice of the user to visualize the videos
    input_type = int(input("Please Select Visualisation Techniques 1 - Opencv, 2 - HOG/HOF Visualization : "))
    if input_type==1:
        # Prompt the user to enter which video to visualize
        while True:
            value = int(input("Do you want to visualise more videos: 1 - Yes, 2 - No: "))
            if value==2:
                print("Visualised the videos")
                break
            elif value==1:
                print(f"Please Select Videos from 1 - {m}")
                query_video = input("Enter the Video name: ")
                found=False
                for video in target_data:
                    if query_video in video:
                        path=f'../dataset/target_videos/{query_video}'
                        found=True
                        break
                if found ==False:
                    for video in non_target_data:
                        if query_video in video:
                            path=f'../dataset/non_target_videos/{query_video}'
                            found=True
                            break
                visualise(path)
    elif input_type == 2:
        BOF_feature_type = int(input("Select a Feature Space: 1 - HOG, 2 - HOF: "))
        BOF_feature_name = 'BOF_HOG' if BOF_feature_type == 1 else 'BOF_HOF'
        # Visualise the videos and prompts the user to 
        while True:
            value = int(input("Do you want to visualise more videos: 1 - Yes, 2 - No: "))
            if value == 2:
                print("Visualised the videos")
                break
            elif value == 1:
                print(f"Please Select Videos from 1 - {m}")
                query_video = input("Enter the Video name: ")
                #Call the function to visualise the video
                Visualize_HoG_HoF(BOF_feature_name, query_video)
#Call the main function
main()
