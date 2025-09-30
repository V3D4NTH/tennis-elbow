import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from feature_engineering import calculate_exercise_specific_features

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2
)

# Load all exercise models
EXERCISE_MODELS = {
    'wrist_extension': torch.load('models/wrist_extension_model.pth'),
    'wrist_flexion': torch.load('models/wrist_flexion_model.pth')
}

for model in EXERCISE_MODELS.values():
    model.eval()

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
            
            # Keep only last 30 frames
            if len(landmarks_buffer) > 30:
                landmarks_buffer.pop(0)
            
            # Detect exercise if we have enough data
            if len(landmarks_buffer) >= 15:
                try:
                    # Detect current exercise
                    detected_exercise = detect_current_exercise(landmarks)
                    
                    # If exercise changed, reset buffer
                    if detected_exercise != current_exercise:
                        current_exercise = detected_exercise
                        landmarks_buffer = landmarks_buffer[-15:]  # Keep some context
                    
                    # Only evaluate if we have a consistent exercise
                    if current_exercise:
                        features = calculate_exercise_specific_features(landmarks_buffer, current_exercise)
                        feature_tensor = torch.tensor(list(features.values()), dtype=torch.float32).unsqueeze(0)
                        
                        # Get prediction from appropriate model
                        with torch.no_grad():
                            output = EXERCISE_MODELS[current_exercise](feature_tensor)
                            _, predicted = torch.max(output.data, 1)
                            
                            # Map predictions to labels
                            labels = ['Good Form', 'Bad Form', 'Mediocre Form']
                            result_text = f"{current_exercise.replace('_', ' ').title()}: {labels[predicted.item()]}"
                            
                            # Display result
                            cv2.putText(frame, result_text, (10, 30),
                                       cv2.QT_TEXT_FLAG_CENTER, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"Error during evaluation: {e}")
        
        # Draw pose landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Rehabilitation Tracker', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_evaluation()