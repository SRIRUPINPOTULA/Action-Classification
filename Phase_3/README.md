# CSE 515 Phase 3 - Group 5a

# Directory Structure:
The Phase-3 directory consists of the following directories

```plaintext
Phase_3/
│
├── README.md                   # Project Overview (this file)
│
├── Code/                       # Source Code for Phase 3
│   └── Util/                   # Source Code to assist Phase 3
│
├── Database/                   # Contains the db and json files
│   ├── Thumbnails_IDs/         # Contains Video Thumbnails named by videoID
│   └── Thumbnails_Names        # Contains Video Thumbnails named by video name
│
├── hmdb51_org/                 # Contains all the original data videos
│
├── Outputs/                    # Contains all the required Outputs from the Tasks
│
└── Report/                     # Contains the report for Phase-3
```

## 1. Code:
This directory consists of all the tasks' codes, including preprocessing and actual tasks.

### Util:

KMeans.py: The python file contains the implementation for KMeans Dimensionality Reduction.

PCA.py: The python file contains the implementation for PCA.

SVD.py: The python file contains the implementation for SVD.

KMeanslatentmodel.py: The python file contains the implementation that gathers all the features of target video and used for calculation of latent models which are used for Task-1b.

services.py: This file contains common variables and functions to extract the latent model for Task 1a.

### Task-0a:

Task_0a.py: Task to find the inherent dimensionality of the feature spaces.

### Task-1:

Task_1_preprocess.py: Gets the thumbnails for all the videos.

Task_1a.py: Implementation of Spectral Clustering.

Task_1b.py: Implementation of KMeans Clustering.

### Task-2:

Task_2.py: Consists that code that implements `KNN` and `SVM` classifiers.

### Task-3:

Task_3a.py: Consists of code that implements LSH for the given input.

Task-3b_preprocess.py: Consists of code that saves the thumbnails for the videos.

Task_3b.py: Consists of code that implements LSH for all the videos

## 2. Database:
This directory consists of the db and json files that are stored as part of all the tasks and the database.

## 3. hmdb51_org:
This directory contains all the original data videos of each category label.

## 4. Outputs:
This directory contains all the required Outputs from the Tasks.

## 5. Report:
This directory contains the report for Phase-3.

<br>
<br>

# How to Run

1. Download, Extract and copy the dataset to hmdb51_org

[Download Link](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

2. Install the required dependencies:
> `pip install -r requirements.txt`

3. Run the Source Code from `Code` directory in order:
> `cd Code`
>
> `python Task_0a.py`
>
> `python Task_1_preprocess.py`
>
> `python Task_1a.py`
>
> `python Task_1b.py`
> 
> `python Task_2.py`
>
> `python Task-3a.py`
>
> `python Task_3_preprocess.py`
>
> `python Task-3b.py`

<br>

# Using the SQLite Database

1. Connect to the database from the CLI
> `sqlite3 database/Phase_3.db`

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
