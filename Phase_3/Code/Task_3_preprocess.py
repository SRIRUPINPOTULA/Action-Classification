import os
import json
import cv2

# Function to extract and save the first frame of each video as a thumbnail
def extract_and_save_thumbnails(json_path, video_directory, output_folder):
    with open(json_path, 'r') as file:
        video_map = json.load(file)

    # Create destination directory
    os.makedirs(output_folder, exist_ok=True)

    for action_dir in os.listdir(video_directory):
        if action_dir == '.gitkeep': #for skipping any irrelevant files in the directory
            continue
        
        action_path = f'{video_directory}/{action_dir}' # Construct the path for the current action directory
        for video in os.listdir(action_path):
            video_path = os.path.join(action_path, video)

            if video in video_map:
                video_id = video_map[video] # Get the corresponding video ID
                thumbnail_path = os.path.join(output_folder, f"{video_id}.jpg") # Path for the thumbnail
                
                if not os.path.exists(thumbnail_path):
                    cap = cv2.VideoCapture(video_path)
                    success, frame = cap.read()
                    if success:
                        cv2.imwrite(thumbnail_path, frame)# If the frame is successfully read, save it as a thumbnail
                    # Release the video capture object
                    cap.release()
    # Completion message
    print("Thumbnails saved successfully.")

json_path = "../Database/videoID.json"
video_directory = "../hmdb51_org"
output_folder = "../Database/Thumbnails_IDs"

extract_and_save_thumbnails(json_path, video_directory, output_folder)
