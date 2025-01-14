# CSE 515 Phase 2 - Group 5

# Directory Structure:
The Phase-2 directory consists of the following directories

```plaintext
Phase_2/
│
├── README.md                   # Project Overview (this file)
│
├── Code/                       # Source Code for Phase 2
│   └── Util/                   # Source Code to assist Phase 2
│
├── database/                   # Contains the db and json files
│
├── dataset/                    # Contains the video data split as target and non target
│   ├── non_target_videos/
│   └── target_videos/
│
├── dataset_stips/              # Contains the video STIPs data split as target and non target
│   ├── non_target_videos/
│   └── target_videos/
│
├── hmdb51_org/                 # Contains all the original data videos
│
├── hmdb51_org_stips/           # Contains all the original videos STIPs data
│
├── Outputs/                    # Contains all the required Outputs from the Tasks
│   └── Task_2/
│
└── Report/                     # Contains the report for Phase-2
```

## 1. Code:
This directory consists of all the tasks' codes, including preprocessing and actual tasks.

### Util:

This Directory Consists of the corrected phase-1 code. 

Task_1.py: This is the phase-1 task1 code that has functions to extract the features from layer3, layer4, avgpool.

Task_2.py: This is the phase-1 task2 code that has functions to extract the HoG and HoF features.

KMeans.py: The python file consists of implementation code for KMeans, used as part of Task-2,3,4.

LDA.py: The python file consists of the LDA implementation that is part of Task-2.

PCA.py: The python file consists implementation of PCA that is used as part of Task-2,3,4.

SVD.py: The python file consists implementation of PCA that is used as part of Task-2,3,4.

Visualize.py: This file contains the implementation of color histograms using HoG and HoF.

### Task-0:

Task_0a.py: Consists code to map the videoID, and extracts all the features.

Task_0b.py: Implements a program to visualize the m most similar videos.

### Task-1:

Task_1_preprocess.py: The values are stored in JSON for each category label this being a preprocessing step.

Task_1.py: The implementation code that lists 'l' most similar labels.

### Task-2:

Task_2.py: Consists that code that implements all the latent features.

### Task-3: 

Task_3.py: Consists of code that implement finds the m similar videos from the latent features and feature spaces.

### Task-4:

Task_4.py: Consists of code that list `m` similar videos from the latent semantics.

### Task-5:

Task_5.py: Incorrect implementation.

### Task-6:

Task_6.py: Incorrect implementation.

## 2. Database:
This directory consists of all the json files that are stored as part of all the tasks.

## 3. Dataset:
This directory consists of two more directories which are target_videos and non_target_videos.

## 4. Dataset Stips:
This directory contains the video STIPs files split into two directories, target_videos and non_target_videos.

## 5. hmdb51_org:

This directory contains all the original data videos of each category label.

## 6. hmdb51_org_stips:
This directory contains all the original videos STIPs data.

<br>
<br>

# How to Run

1. Download, Extract and copy the dataset to hmdb51_org and hmdb51_org_stips

[Download Link](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

2. Install the required dependencies:
> `pip install -r requirements.txt`

3. Run Task 0a only on first time setup:
> `python Code/Task_0a.py`

4. Run the Source Code from `Code` directory in order:
> `cd Code`
>
> `python Task_0b.py`
>
> `python Task_1_preprocess.py`
> 
> `python Task_1.py`
>
> `python Task_2.py`
>
> `python Task_3.py`
>
> `python Task_4.py`

<br>

# Using the SQLite Database

1. Connect to the database from the CLI
> `sqlite3 database/Phase_2.db`

2. List the tables in the db
> `.tables`

3. View the table Schema
> `.schema data`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or

>`PRAGMA table_info(data)`

4. Use SQL to query db
> `select * from data limit 5;`

5. Other Resources

- [SQLite Browser](http://sqlitebrowser.org/)
- [SQLite Viewer](https://inloop.github.io/sqlite-viewer/)
- [spatialite-gui](https://www.gaia-gis.it/fossil/spatialite_gui/index)

