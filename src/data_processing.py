import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import re

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2
)

def get_exercise_category(folder_name):
    """Extract exercise category from folder name"""
    if 'Wrist_Extension' in folder_name:
        return 'wrist_extension'
    elif 'Wrist_Flexion' in folder_name:
        return 'wrist_flexion'
    return 'unknown'

def process_folder_structure(raw_videos_dir, processed_data_dir):
    """Process videos organized in nested folders"""
    for root, dirs, files in os.walk(raw_videos_dir):
        for filename in files:
            if filename.endswith('.mp4'):
                video_path = os.path.join(root, filename)
                
                # Determine exercise category from parent folder
                parent_folder = os.path.basename(root)
                exercise_category = get_exercise_category(parent_folder)
                
                if exercise_category == 'unknown':
                    continue
                
                # Create destination path
                dest_dir = os.path.join(processed_data_dir, exercise_category, parent_folder)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Process video
                cap = cv2.VideoCapture(video_path)
                landmarks_list = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    
                    if results.pose_landmarks:
                        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                        landmarks_list.append(landmarks)
                
                cap.release()
                
                # Save as pickle
                output_filename = filename.replace('.mp4', '.pkl')
                output_path = os.path.join(dest_dir, output_filename)
                with open(output_path, 'wb') as f:
                    pickle.dump(landmarks_list, f)
                
                print(f"Processed {video_path} -> saved to {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_videos_dir', required=True)
    parser.add_argument('--processed_data_dir', required=True)
    args = parser.parse_args()
    
    process_folder_structure(args.raw_videos_dir, args.processed_data_dir)