import os
import cv2

def extract_and_save_thumbnails(video_directory, output_folder):

    # Create destination directory
    os.makedirs(output_folder, exist_ok=True)

    for action_dir in os.listdir(video_directory):
        if action_dir == '.gitkeep':
            continue
        
        action_path = f'{video_directory}/{action_dir}'
        for video in os.listdir(action_path):
            video_path = os.path.join(action_path, video)

            thumbnail_path = os.path.join(output_folder, f"{video}.jpg")
            
            if not os.path.exists(thumbnail_path):
                cap = cv2.VideoCapture(video_path)
                success, frame = cap.read()
                if success:
                    cv2.imwrite(thumbnail_path, frame)
                cap.release()

    print("Thumbnails saved successfully.")

video_directory = "../hmdb51_org"
output_folder = "../Database/Thumbnails_Names"

extract_and_save_thumbnails(video_directory, output_folder)
