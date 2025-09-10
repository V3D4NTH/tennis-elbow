import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import math
import traceback

# Exercise configuration
EXERCISE_TYPES = ['wrist_extension', 'wrist_flexion']

# Exercise configuration
EXERCISE_CONFIG = {
    'wrist_extension': {
        'key_joints': {'elbow': 13, 'wrist': 15},
        'primary_metric': 'elbow_flexion',
        'secondary_metrics': ['wrist_deviation', 'movement_smoothness']
    },
    'wrist_flexion': {
        'key_joints': {'elbow': 13, 'wrist': 15},
        'primary_metric': 'wrist_flexion_angle',
        'secondary_metrics': ['elbow_stability', 'range_of_motion']
    }
}

def safe_calculate_angle(v1, v2):
    """Safely calculate angle between two vectors"""
    try:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0 and norm_v2 > 0:
            cos_theta = dot_product / (norm_v1 * norm_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return math.acos(cos_theta) * 180 / math.pi
        return 0.0
    except Exception as e:
        print(f"Error calculating angle: {e}")
        traceback.print_exc()
        return 0.0

def calculate_exercise_specific_features(landmarks_sequence, exercise_type):
    """Calculate biomechanical features tailored to specific exercises"""
    try:
        config = EXERCISE_CONFIG[exercise_type]
        features = {}
        
        # Primary metric calculation
        if config['primary_metric'] == 'elbow_flexion':
            try:
                # Check if we have enough landmarks
                if len(landmarks_sequence) < 12:  # Need at least 12 landmarks for shoulder, elbow, wrist
                    print(f"Warning: Insufficient landmarks for elbow_flexion calculation. Got {len(landmarks_sequence)} landmarks.")
                    features['elbow_flexion'] = 0.0
                else:
                    shoulder = np.array([
                        landmarks_sequence[11][0],
                        landmarks_sequence[11][1],
                        landmarks_sequence[11][2]
                    ], dtype=np.float32)
                    
                    elbow = np.array([
                        landmarks_sequence[config['key_joints']['elbow']][0],
                        landmarks_sequence[config['key_joints']['elbow']][1],
                        landmarks_sequence[config['key_joints']['elbow']][2]
                    ], dtype=np.float32)
                    
                    wrist = np.array([
                        landmarks_sequence[config['key_joints']['wrist']][0],
                        landmarks_sequence[config['key_joints']['wrist']][1],
                        landmarks_sequence[config['key_joints']['wrist']][2]
                    ], dtype=np.float32)
                    
                    # Vector from elbow to wrist
                    v1 = wrist - elbow
                    
                    # Reference vector (vertical)
                    v2 = np.array([0, -1, 0], dtype=np.float32)  # Downward reference vector
                    
                    features['elbow_flexion'] = safe_calculate_angle(v1, v2)
            except IndexError as e:
                print(f"Error in elbow_flexion calculation: {e}")
                traceback.print_exc()
                features['elbow_flexion'] = 0.0
        
        elif config['primary_metric'] == 'wrist_flexion_angle':
            try:
                # Check if we have enough landmarks
                if len(landmarks_sequence) < config['key_joints']['elbow'] + 1:  # Need elbow and wrist
                    print(f"Warning: Insufficient landmarks for wrist_flexion_angle calculation. Got {len(landmarks_sequence)} landmarks.")
                    features['wrist_flexion_angle'] = 0.0
                else:
                    elbow = np.array([
                        landmarks_sequence[config['key_joints']['elbow']][0],
                        landmarks_sequence[config['key_joints']['elbow']][1],
                        landmarks_sequence[config['key_joints']['elbow']][2]
                    ], dtype=np.float32)
                    
                    wrist = np.array([
                        landmarks_sequence[config['key_joints']['wrist']][0],
                        landmarks_sequence[config['key_joints']['wrist']][1],
                        landmarks_sequence[config['key_joints']['wrist']][2]
                    ], dtype=np.float32)
                    
                    # Vector from elbow to wrist
                    v1 = wrist - elbow
                    
                    # Reference vector (vertical)
                    v2 = np.array([0, -1, 0], dtype=np.float32)  # Downward reference vector
                    
                    features['wrist_flexion_angle'] = safe_calculate_angle(v1, v2)
            except IndexError as e:
                print(f"Error in wrist_flexion_angle calculation: {e}")
                traceback.print_exc()
                features['wrist_flexion_angle'] = 0.0
        
        # Secondary metrics
        for metric in config['secondary_metrics']:
            try:
                if metric == 'wrist_deviation':
                    features['wrist_deviation'] = calculate_wrist_deviation(
                        landmarks_sequence, 
                        config['key_joints']['elbow'], 
                        config['key_joints']['wrist']
                    )
                elif metric == 'movement_smoothness':
                    features['movement_smoothness'] = calculate_movement_smoothness(landmarks_sequence)
                elif metric == 'elbow_stability':
                    features['elbow_stability'] = calculate_elbow_stability(
                        landmarks_sequence, 
                        config['key_joints']['elbow']
                    )
                elif metric == 'range_of_motion':
                    features['range_of_motion'] = calculate_range_of_motion(
                        landmarks_sequence, 
                        config['key_joints']['elbow'], 
                        config['key_joints']['wrist']
                    )
            except IndexError as e:
                print(f"Error calculating secondary metric {metric}: {e}")
                traceback.print_exc()
                features[metric] = 0.0
        
        return features
    except Exception as e:
        print(f"Error in calculate_exercise_specific_features: {e}")
        traceback.print_exc()
        return {}

def calculate_wrist_deviation(landmarks_sequence, elbow_idx, wrist_idx):
    """Calculate wrist deviation angle with error handling"""
    try:
        # Check if we have enough landmarks
        if len(landmarks_sequence) <= max(elbow_idx, wrist_idx):
            print(f"Warning: Insufficient landmarks for wrist_deviation calculation. Got {len(landmarks_sequence)} landmarks.")
            return 0.0
        
        elbow = np.array([
            landmarks_sequence[elbow_idx][0],
            landmarks_sequence[elbow_idx][1],
            landmarks_sequence[elbow_idx][2]
        ], dtype=np.float32)
        
        wrist = np.array([
            landmarks_sequence[wrist_idx][0],
            landmarks_sequence[wrist_idx][1],
            landmarks_sequence[wrist_idx][2]
        ], dtype=np.float32)
        
        # Vector from elbow to wrist
        v1 = wrist - elbow
        
        # Reference vector (vertical)
        v2 = np.array([0, 1, 0], dtype=np.float32)  # Pointing upward
        
        return safe_calculate_angle(v1, v2)
    except IndexError as e:
        print(f"Error in calculate_wrist_deviation: {e}")
        traceback.print_exc()
        return 0.0

def calculate_movement_smoothness(landmarks_sequence):
    """Calculate movement smoothness based on landmark trajectory"""
    try:
        positions = [landmarks[15][:2] for landmarks in landmarks_sequence]
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        jerk = np.mean(np.abs(accelerations))
        return float(jerk)
    except Exception as e:
        print(f"Error in calculate_movement_smoothness: {e}")
        traceback.print_exc()
        return 0.0

def calculate_elbow_stability(landmarks_sequence, elbow_idx):
    """Calculate elbow stability by analyzing variance in joint position"""
    try:
        elbow_positions = [landmarks[elbow_idx][:3] for landmarks in landmarks_sequence]
        elbow_positions = np.array(elbow_positions)
        
        # Calculate standard deviation of elbow position
        std_deviation = np.std(elbow_positions, axis=0)
        overall_stability = np.mean(std_deviation)
        
        return float(overall_stability)
    except Exception as e:
        print(f"Error in calculate_elbow_stability: {e}")
        traceback.print_exc()
        return 0.0

def calculate_range_of_motion(landmarks_sequence, elbow_idx, wrist_idx):
    """Calculate range of motion between elbow and wrist"""
    try:
        elbow_positions = [landmarks[elbow_idx][:3] for landmarks in landmarks_sequence]
        wrist_positions = [landmarks[wrist_idx][:3] for landmarks in landmarks_sequence]
        
        # Calculate distances between elbow and wrist
        distances = [np.linalg.norm(np.array(elbow) - np.array(wrist)) 
                     for elbow, wrist in zip(elbow_positions, wrist_positions)]
        
        # Return max distance minus min distance
        return float(max(distances) - min(distances))
    except Exception as e:
        print(f"Error in calculate_range_of_motion: {e}")
        traceback.print_exc()
        return 0.0

class ExerciseDataset(Dataset):
    def __init__(self, data_dir, exercise_type):
        self.exercise_type = exercise_type
        self.data_paths = []
        self.labels = []
        
        exercise_dir = os.path.join(data_dir, exercise_type)
        if not os.path.exists(exercise_dir):
            raise FileNotFoundError(f"No data found for {exercise_type}")
        
        for subfolder in os.listdir(exercise_dir):
            subfolder_path = os.path.join(exercise_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.endswith('.pkl'):
                        self.data_paths.append(os.path.join(subfolder_path, filename))
                        
                        # Extract label from filename
                        if 'good' in filename.lower():
                            self.labels.append(0)  # Good form
                        elif 'bad' in filename.lower():
                            self.labels.append(1)  # Bad form
                        elif 'mediocre' in filename.lower():
                            self.labels.append(2)  # Mediocre form
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        try:
            with open(self.data_paths[idx], 'rb') as f:
                landmarks_sequence = pickle.load(f)
            
            features = calculate_exercise_specific_features(landmarks_sequence, self.exercise_type)
            feature_vector = list(features.values())
            
            # Ensure consistent feature vector length
            expected_length = len(EXERCISE_CONFIG[self.exercise_type]['secondary_metrics']) + 1
            if len(feature_vector) != expected_length:
                print(f"Warning: Inconsistent feature vector length at index {idx}. Expected {expected_length}, got {len(feature_vector)}. Filling with zeros.")
                # Pad with zeros if needed
                if len(feature_vector) < expected_length:
                    feature_vector.extend([0.0] * (expected_length - len(feature_vector)))
                else:
                    feature_vector = feature_vector[:expected_length]
            
            return torch.tensor(feature_vector, dtype=torch.float32), self.labels[idx]
        except Exception as e:
            print(f"Error loading data for item {idx}: {e}")
            traceback.print_exc()
            # Return dummy data to keep the program running
            return torch.zeros(expected_length, dtype=torch.float32), 0

def train_model_for_exercise(exercise_type, data_dir):
    """Train model for a specific exercise type"""
    try:
        # Load dataset
        dataset = ExerciseDataset(data_dir, exercise_type)
        if len(dataset) == 0:
            print(f"No data available for {exercise_type}. Skipping...")
            return None
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        
        # Initialize model (simple CNN for demonstration)
        num_features = len(EXERCISE_CONFIG[exercise_type]['secondary_metrics']) + 1
        model = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        best_accuracy = 0
        for epoch in range(20):  # Reduced epochs for quicker testing
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'{exercise_type} - Epoch [{epoch+1}/20], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), f'models/{exercise_type}_model.pth')
        
        print(f"Successfully trained {exercise_type} model with {best_accuracy:.2f}% accuracy")
        return model
        
    except Exception as e:
        print(f"Error training {exercise_type} model: {e}")
        traceback.print_exc()
        return None

def auto_train_all_exercises(data_dir):
    """Automatically train models for all available exercises"""
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Find all exercises with data
    available_exercises = []
    for exercise in EXERCISE_TYPES:
        exercise_dir = os.path.join(data_dir, exercise)
        if os.path.exists(exercise_dir) and len(os.listdir(exercise_dir)) > 0:
            available_exercises.append(exercise)
    
    if not available_exercises:
        print("No exercises with data found. Please check your data directory.")
        return
    
    print(f"Found {len(available_exercises)} exercises with data: {', '.join(available_exercises)}")
    
    # Train models for all available exercises
    for exercise in available_exercises:
        print(f"\nTraining model for {exercise}...")
        train_model_for_exercise(exercise, data_dir)

if __name__ == '__main__':
    # Use the specific path you provided
    auto_train_all_exercises('C:\\Users\\itsth\\Desktop\\skill issue\\ugh\\capstone\\tennis_elbow_rehab1\\data\\processed_data')