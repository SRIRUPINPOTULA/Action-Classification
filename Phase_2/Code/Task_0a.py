import os
import shutil
import json

# import SQL library
import sqlite3

# import the python libraries
import numpy as np

# import functions to generate feature spaces
from Util.Task_1 import layer3_feature
from Util.Task_1 import layer4_feature
from Util.Task_1 import avgpool_feature
from Util.Task_2 import create_cluster_representatives
from Util.Task_2 import get_HoG_HoF_features

target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke', 
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']

# Dictionary to map video name to videoID
videoID = {}
# Dictionary to map video name to Action
actionmap = {}


# Connect to the database
connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()

# Create database tables
create_data_table = """CREATE TABLE IF NOT EXISTS data (
                    videoID INTEGER PRIMARY KEY,
                    Video_Name VARCHAR(250),
                    Layer_3 VARCHAR(4000),
                    Layer_4 VARCHAR(4000),
                    AvgPool VARCHAR(4000),
                    BOF_HOG VARCHAR(4000),
                    BOF_HOF VARCHAR(4000),
                    Action_Label VARCHAR(20));"""

                    # """CREATE TABLE IF NOT EXISTS data (
                    # videoID INTEGER PRIMARY KEY,
                    # Video_Name TEXT NOT NULL,
                    # Layer_3 TEXT,
                    # Layer_4 TEXT,
                    # AvgPool TEXT,
                    # BOF_HOG TEXT,
                    # BOF_HOF TEXT,
                    # Action_Label TEXT);"""

# TODO: create table for P1_Task3
# create_color_hist_table = """CREATE TABLE IF NOT EXISTS color_hist_data ();"""

c.execute(create_data_table)
# c.execute(create_color_hist_table)

#Mapping the video id to all the target and non target videos
def create_video_id():
    counter = 0
    target_path = '../dataset/target_videos'
    non_target_path = '../dataset/non_target_videos'
    for file in os.listdir(target_path):
        videoID[file] = counter
        counter += 1
    for file in os.listdir(non_target_path):
        videoID[file] = counter
        counter += 1
    
    # JSON is used as a backup to data
    with open('../database/videoID.json', 'w') as f:
        json.dump(videoID, f, indent=4)

# Video Category Map for all the even numbered target videos
def video_category_map():
    dataset = '../hmdb51_org'
    for action in os.listdir(dataset):
        action_path = os.path.join(dataset, action)
        if action in target_videos:
            for file in os.listdir(action_path):
                if videoID[file]%2 == 0:
                    a = {}
                    a[file] = action
                    actionmap.update(a)
    
    # JSON is used as a backup to data
    with open('../database/category_map.json', 'w') as f:
        json.dump(actionmap, f, indent=4)

#Extracting Features for all the target videos
def target_videos_features():
    target_path = '../dataset/target_videos'
    target_feature = []
    for target_videos in os.listdir(target_path):
        video_path = os.path.join(target_path, target_videos)

        video_features_map = {}
        video_features = []
        # Extract the feature as a layer3
        layer_3 = layer3_feature(video_path)
        video_features.append(layer_3)
        # Extract the feature as a layer4
        layer_4 = layer4_feature(video_path)
        video_features.append(layer_4)
        # Extract the feature as a avgpool
        avgPool = avgpool_feature(video_path)
        #Append to a list
        video_features.append(avgPool)
        # Gather features for all the HoG, HoF ad append to a list
        bof_HOG, bof_HOF = get_HoG_HoF_features('../dataset_stips/target_videos/' + target_videos + '.txt')
        video_features.append(bof_HOG.tolist())
        video_features.append(bof_HOF.tolist())
        # TODO: Add Task3 Features

        video_features_map[target_videos] = video_features
        target_feature.append(video_features_map)
        #Insert into database all the features
        if videoID[target_videos]%2 == 0:
            video_insert = f"INSERT INTO data VALUES({videoID[target_videos]}, '{target_videos}', '{layer_3}', '{layer_4}', '{avgPool}', '{bof_HOG}', '{bof_HOF}', '{actionmap[target_videos]}');"
        else:
            video_insert = f"INSERT INTO data VALUES({videoID[target_videos]}, '{target_videos}', '{layer_3}', '{layer_4}', '{avgPool}', '{bof_HOG}', '{bof_HOF}', NULL);"
        c.execute(video_insert)
    # Dump all the target features
    with open('../database/total_target_features.json', 'w') as f:
        json.dump(target_feature, f, indent=4)

# Extract features for all the non target videos
def non_target_videos_features():
    non_target_path = '../dataset/non_target_videos'
    non_target_feature = []
    for non_target_videos in os.listdir(non_target_path):
        video_path = os.path.join(non_target_path, non_target_videos)

        video_features_map = {}
        video_features = []
        # Extract the feature as a layer3
        layer_3 = layer3_feature(video_path)
        video_features.append(layer_3)
        # Extract the feature as a layer4
        layer_4 = layer4_feature(video_path)
        video_features.append(layer_4)
        # Extract the feature as a avgpool
        avgPool = avgpool_feature(video_path)
        video_features.append(avgPool)
        # Extract the feature as a HoG and HoF
        bof_HOG, bof_HOF = get_HoG_HoF_features('../dataset_stips/non_target_videos/' + non_target_videos + '.txt')
        video_features.append(bof_HOG.tolist())
        video_features.append(bof_HOF.tolist())
        # TODO: Add Task3 Features

        video_features_map[non_target_videos] = video_features
        non_target_feature.append(video_features_map)
        # Insert into Database
        video_insert = f"INSERT INTO data VALUES({videoID[non_target_videos]}, '{non_target_videos}', '{layer_3}', '{layer_4}', '{avgPool}', '{bof_HOG}', '{bof_HOF}', NULL);"
        c.execute(video_insert)
    # Dumping all the videos in the non target video features
    with open('../database/total_non_target_features.json', 'w') as f:
        json.dump(non_target_feature, f, indent=4)


# Move all videos from 'hmdb51_org' to 'dataset/target_videos' and 'dataset/non_target_videos'
def move_videos():
    path = '../hmdb51_org'
    #Check if already the directory exists if so, delete it to create directory for target videos
    target_path = os.listdir('../dataset/target_videos')
    for file in target_path:
        full_path = os.path.join('../dataset/target_videos', file)
        if os.path.isfile(full_path):
            os.remove(full_path)
    #Check if already the directory exists if so, delete it to create directory for non target videos
    non_target_path = os.listdir('../dataset/non_target_videos')
    for file in non_target_path:
        full_path = os.path.join('../dataset/non_target_videos', file)
        if os.path.isfile(full_path):
            os.remove(full_path)
    #Transfer all the videos to their respective directories
    for video_dir in os.listdir(path):
        if video_dir in target_videos:
            video_path = f'{path}/{video_dir}'
            destination_dir = '../dataset/target_videos'
            for all_videos in os.listdir(video_path):
                file_name = os.path.join(video_path, all_videos)
                shutil.copy(file_name, destination_dir)
        elif video_dir == '.DS_Store' or video_dir == '.gitkeep':
            continue
        else:
            video_path = f'{path}/{video_dir}'
            destination_dir = '../dataset/non_target_videos'
            for all_videos in os.listdir(video_path):
                file_name = os.path.join(video_path, all_videos)
                shutil.copy(file_name, destination_dir)


# Move all video STIPs from 'hmdb51_org_stips' to 'dataset_stips/target_videos' and 'dataset_stips/non_target_videos'
def move_stips():
    path = '../hmdb51_org_stips'
    # Check if already the directory exists if so, delete it to create directory for target videos
    target_path = os.listdir('../dataset_stips/target_videos')
    for file in target_path:
        full_path = os.path.join('../dataset_stips/target_videos', file)
        if os.path.isfile(full_path):
            os.remove(full_path)
    # Check if already the directory exists if so, delete it to create directory for non target videos
    non_target_path = os.listdir('../dataset_stips/non_target_videos')
    for file in non_target_path:
        full_path = os.path.join('../dataset_stips/non_target_videos', file)
        if os.path.isfile(full_path):
            os.remove(full_path)
    # Transfer all the videos to their respective directories
    for video_dir in os.listdir(path):
        if video_dir in target_videos:
            video_path = f'{path}/{video_dir}'
            destination_dir = '../dataset_stips/target_videos'
            for all_videos in os.listdir(video_path):
                file_name = os.path.join(video_path, all_videos)
                shutil.copy(file_name, destination_dir)
        elif video_dir == '.DS_Store' or video_dir == '.gitkeep':
            continue
        else:
            video_path = f'{path}/{video_dir}'
            destination_dir = '../dataset_stips/non_target_videos'
            for all_videos in os.listdir(video_path):
                file_name = os.path.join(video_path, all_videos)
                shutil.copy(file_name, destination_dir)

#Calling the functions
move_videos()
move_stips()

create_cluster_representatives()
create_video_id()
video_category_map()
target_videos_features()
non_target_videos_features()

# commit the database changes
connection.commit()

# close the database connection
connection.close()
