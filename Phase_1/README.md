# CSE 515 - Multimedia and Web Databases 

# Title: Action Classification

# Data

The videos are divided into target and non_target videos. So, during the execution of the programs the input video file names are based on the location of the videos. 

`Example Usage: sample_input_1 = '/Users/srirupin/Downloads/target/cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi'`

# Repo Directory Structure:
# 1.Code Directory: 

## 1.1 Task -1: 
In directory task-1, `Task-1.ipynb` file has been added, corresponding to the code and output for Task-1. In this Python file itself, the outputs of the sample inputs have been printed.

## 1.2 Task -2:
In directory task-2, `Task 2.ipynb` file has been added, corresponding to the code and output for Task-2.

## 1.3 Task -3: 

`Task_3_csv_method.ipynb`, `Task_3_histogram_png_plots.ipynb` are added which are used to obtain vector forms of the histograms in the form of csv file while the `Task_3_histogram_png_plots.ipyn`b is used to get the png output of histograms. csv output files are smaller, easier to compute hence used for the entire phase. PNG files are more visual but size is drastically huge and difficult to use them for computation. Output\task-3\Task_4_using_3 is the folder which consists of the output csvs. They have been zipped for easier uploading as well as downloading.

## 1.4 Task-4:  
In directory task-4, `Task-4.ipynb` file was added, corresponding to the code used to generate the output for all the videos of target videos. The outputs are exported to JSON which is part of `Output\Task-4\Task-1_Output`.

## 1.5 Task-5:  
In directory task-5, `Task-5.ipynb` file has been added that measures the `m=10` similar videos for the video file given using the JSON of Task-4 and using the logic of Task-1. The nearest search videos for the sample inputs have been printed there along with the distance. 

In directory task-5, `Task_5(using task 3).ipynb` file has been added that measures m similar videos using the logic of task-3.

In directory task-5, `Task 5_2.ipynb` file has been added that measures m similar videos using the logic of task-2.

# 2. Output:

## 2.1 Task-1 Output: 
### 2.1.1 Features of Target Videos:
In the output directory, the directory `Task-4` contains the output of the code made for `Task-1_Output`. Each of the json files represents the target actions. The mapping of them to the target videos are listed below:

Output of action type sword => `sword_output.json`

Output of action type cartwheel => `cartwheel_output.json`

Output of action type drink => `drink_output.json`

Output of action type ride_bike => `ridebikeoutput.json`

### 2.1.1 Output for Input Videos:

In the output directory, the directory `Task-5` contains the output of the code made for task-1. The output for the sample videa=os are located at `output_task1.json`. 

## 2.2 Task-2 Output: 
### 2.2.1 Features of Target Video:
In the output directory, the directory `Task-4` contains the output of the code made for `Task-2_Output`. This directory consists of a directory of all the actions that are listed as target videos. And in each action directory for each input video, the output is stored as JSON.

### 2.2.2 Output for Input Videos:

In the directory, `Task-5` the folder `Output_Task_2` contains the HoF and HoG for the videos. The output files for HoG and HoF which contains the HoG and HoF for two input videos.

## 2.3 Task-3 Output:
Using the csv method (Task_3_csv_method.ipynb), `Output/Task-4/Task_4_using_3` has 6 folders, of each action type. Each folder consists of csv file in the nomemclature format of "{video_name}histogram_frame{i}.csv". "i" is the frame number. The csv file contains the histogram vectors made using "r"=4 and "n"=12, "r" and "n" are resolution and bin count respectively.`Output/Task-4/Task_4_using_3/Histogram` plots for sample input videos folder contains the histograms for the 4 input sample videos.  Using csv method, `Output/Task-5/Output_task3/Task-5`(using task 3).docx is a document which has the screenshots which we get for the particular task. The name of "m" most similar videos are displayed along with their similarity scores.
