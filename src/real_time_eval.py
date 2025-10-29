import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import math
import os

# Import the CNN model from model_training
from model_training import TemporalCNN

# Detect available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2
)

# Load all exercise models
def load_models():
    """Load trained CNN models for each exercise"""
    models = {}
    if not os.path.exists('models'):
        print("Models directory not found. Please train models first using: python src/model_training.py")
        return models
    
    for exercise in ['wrist_extension', 'wrist_flexion']:
        model_path = f'models/{exercise}_model.pth'
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}. Please train models first.")
            continue
        
        # Try loading with 99 channels (3D) first, fall back to 66 (2D)
        for in_channels in [99, 66]:
            try:
                model = TemporalCNN(in_channels=in_channels, num_classes=3)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)
                model.eval()
                models[exercise] = model
                print(f"Loaded {exercise} model successfully with {in_channels} input channels")
                break
            except Exception as e:
                if in_channels == 66:  # Last attempt failed
                    print(f"Error loading {exercise} model: {e}")
                continue
    
    if not models:
        print("No models loaded. Train models using: python src/model_training.py")
    return models

EXERCISE_MODELS = load_models()

def preprocess_landmarks(landmarks_buffer, target_length=60):
    """Preprocess landmarks same as training data"""
    # Convert to numpy array
    landmarks_array = np.array(landmarks_buffer, dtype=np.float32)
    
    # Handle different shapes
    if len(landmarks_array.shape) == 3:
        # Shape: (seq_len, num_landmarks, num_coords)
        seq_len, num_landmarks, num_coords = landmarks_array.shape
        # Flatten and transpose
        landmarks_flat = landmarks_array.reshape(seq_len, num_landmarks * num_coords).T
    elif len(landmarks_array.shape) == 2:
        # Already flattened
        landmarks_flat = landmarks_array.T
    else:
        return None
    
    # Pad or truncate to target length
    current_length = landmarks_flat.shape[1]
    
    if current_length < target_length:
        # Pad with zeros
        padding = np.zeros((landmarks_flat.shape[0], target_length - current_length), dtype=np.float32)
        landmarks_flat = np.concatenate([landmarks_flat, padding], axis=1)
    elif current_length > target_length:
        # Truncate
        landmarks_flat = landmarks_flat[:, :target_length]
    
    return landmarks_flat

def detect_current_exercise(landmarks):
    """Detect which exercise is currently being performed"""
    elbow = landmarks[13]
    wrist = landmarks[15]
    
    # Calculate wrist-elbow angle
    v1 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1], wrist[2] - elbow[2]])
    v2 = np.array([0, -1, 0])  # Downward reference
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    
    if norm_v1 > 0:
        cos_theta = dot_product / norm_v1
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = math.acos(cos_theta) * 180 / math.pi
        
        # Heuristic for exercise detection
        if angle < 45:  # More vertical - wrist flexion
            return 'wrist_flexion'
        else:  # More horizontal - wrist extension
            return 'wrist_extension'
    
    return 'wrist_extension'  # Default

def real_time_evaluation():
    """Perform real-time evaluation for multiple exercises"""
    cap = cv2.VideoCapture(0)
    landmarks_buffer = []
    current_exercise = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            landmarks_buffer.append(landmarks)
            
            # Keep only last 60 frames (matching training target_length)
            if len(landmarks_buffer) > 60:
                landmarks_buffer.pop(0)
            
            # Detect exercise if we have enough data
            if len(landmarks_buffer) >= 30:  # Need at least 30 frames for good prediction
                try:
                    # Detect current exercise
                    detected_exercise = detect_current_exercise(landmarks)
                    
                    # If exercise changed, reset buffer
                    if detected_exercise != current_exercise:
                        current_exercise = detected_exercise
                        landmarks_buffer = landmarks_buffer[-30:]  # Keep some context
                    
                    # Only evaluate if we have a model for this exercise
                    if current_exercise and current_exercise in EXERCISE_MODELS:
                        # Preprocess landmarks same as training
                        landmarks_tensor = preprocess_landmarks(landmarks_buffer)
                        
                        if landmarks_tensor is not None:
                            # Convert to torch tensor and add batch dimension
                            input_tensor = torch.tensor(landmarks_tensor, dtype=torch.float32).unsqueeze(0).to(device)
                            
                            # Get prediction from appropriate model
                            with torch.no_grad():
                                output = EXERCISE_MODELS[current_exercise](input_tensor)
                                _, predicted = torch.max(output.data, 1)
                                
                                # Map predictions to labels
                                labels = ['Good Form', 'Bad Form', 'Mediocre Form']
                                result_text = f"{current_exercise.replace('_', ' ').title()}: {labels[predicted.item()]}"
                                
                                # Display result
                                cv2.putText(frame, result_text, (10, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Draw pose landmarks
        mp_drawing = mp.solutions.drawing_utils
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Rehabilitation Tracker', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_evaluation()
