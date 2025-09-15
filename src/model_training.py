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

# Configuration variables
EPOCHS = 20  # Control number of epochs

# Global error logger
ERROR_LOG = {
    'count': 0,
    'reasons': {},
    'incompatible_data_points': 0
}

def log_error(reason):
    ERROR_LOG['count'] += 1
    ERROR_LOG['reasons'][reason] = ERROR_LOG['reasons'].get(reason, 0) + 1

def safe_calculate_angle(v1, v2):
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
        log_error(f"Error calculating angle: {str(e)}")
        return 0.0

def calculate_exercise_specific_features(landmarks_sequence, exercise_type):
    try:
        config = EXERCISE_CONFIG[exercise_type]
        features = {}
        
        # Primary metric calculation
        if config['primary_metric'] == 'elbow_flexion':
            try:
                if len(landmarks_sequence) < 12:
                    log_error(f"Insufficient landmarks for elbow_flexion: got {len(landmarks_sequence)}, required 12")
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
                    
                    v1 = wrist - elbow
                    v2 = np.array([0, -1, 0], dtype=np.float32)  # Downward reference vector
                    features['elbow_flexion'] = safe_calculate_angle(v1, v2)
            except IndexError as e:
                log_error(f"IndexError in elbow_flexion calculation: {str(e)}")
                features['elbow_flexion'] = 0.0
        
        elif config['primary_metric'] == 'wrist_flexion_angle':
            try:
                if len(landmarks_sequence) < config['key_joints']['elbow'] + 1:
                    log_error(f"Insufficient landmarks for wrist_flexion_angle: got {len(landmarks_sequence)}, required {config['key_joints']['elbow'] + 1}")
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
                    
                    v1 = wrist - elbow
                    v2 = np.array([0, -1, 0], dtype=np.float32)  # Downward reference vector
                    features['wrist_flexion_angle'] = safe_calculate_angle(v1, v2)
            except IndexError as e:
                log_error(f"IndexError in wrist_flexion_angle calculation: {str(e)}")
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
            except Exception as e:
                log_error(f"Error calculating secondary metric {metric}: {str(e)}")
                features[metric] = 0.0
        
        return features
    except Exception as e:
        log_error(f"Error in calculate_exercise_specific_features: {str(e)}")
        return {}

def calculate_wrist_deviation(landmarks_sequence, elbow_idx, wrist_idx):
    try:
        if len(landmarks_sequence) <= max(elbow_idx, wrist_idx):
            log_error(f"Insufficient landmarks for wrist_deviation: got {len(landmarks_sequence)}, required {max(elbow_idx, wrist_idx) + 1}")
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
        
        v1 = wrist - elbow
        v2 = np.array([0, 1, 0], dtype=np.float32)  # Pointing upward
        return safe_calculate_angle(v1, v2)
    except IndexError as e:
        log_error(f"IndexError in calculate_wrist_deviation: {str(e)}")
        return 0.0

def calculate_movement_smoothness(landmarks_sequence):
    try:
        positions = [landmarks[15][:2] for landmarks in landmarks_sequence]
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        jerk = np.mean(np.abs(accelerations))
        return float(jerk)
    except Exception as e:
        log_error(f"Error in calculate_movement_smoothness: {str(e)}")
        return 0.0

def calculate_elbow_stability(landmarks_sequence, elbow_idx):
    try:
        elbow_positions = [landmarks[elbow_idx][:3] for landmarks in landmarks_sequence]
        elbow_positions = np.array(elbow_positions)
        
        std_deviation = np.std(elbow_positions, axis=0)
        overall_stability = np.mean(std_deviation)
        
        return float(overall_stability)
    except Exception as e:
        log_error(f"Error in calculate_elbow_stability: {str(e)}")
        return 0.0

def calculate_range_of_motion(landmarks_sequence, elbow_idx, wrist_idx):
    try:
        elbow_positions = [landmarks[elbow_idx][:3] for landmarks in landmarks_sequence]
        wrist_positions = [landmarks[wrist_idx][:3] for landmarks in landmarks_sequence]
        
        distances = [np.linalg.norm(np.array(elbow) - np.array(wrist)) 
                     for elbow, wrist in zip(elbow_positions, wrist_positions)]
        
        return float(max(distances) - min(distances))
    except Exception as e:
        log_error(f"Error in calculate_range_of_motion: {str(e)}")
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
                log_error(f"Inconsistent feature vector length at index {idx}: expected {expected_length}, got {len(feature_vector)}")
                if len(feature_vector) < expected_length:
                    feature_vector.extend([0.0] * (expected_length - len(feature_vector)))
                else:
                    feature_vector = feature_vector[:expected_length]
            
            return torch.tensor(feature_vector, dtype=torch.float32), self.labels[idx]
        except Exception as e:
            log_error(f"Error loading data for item {idx}: {str(e)}")
            expected_length = len(EXERCISE_CONFIG[self.exercise_type]['secondary_metrics']) + 1
            return torch.zeros(expected_length, dtype=torch.float32), 0

def create_model(model_type, num_features):
    """Create specified model type"""
    if model_type == "neural_net":
        return nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    elif model_type == "lstm":
        return nn.LSTM(input_size=num_features, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)
    elif model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_neural_network(exercise_type, train_loader, val_loader, num_features):
    """Train neural network model"""
    model = create_model("neural_net", num_features)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_accuracy = 0
    for epoch in range(EPOCHS):
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
        print(f'{exercise_type} - Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'models/{exercise_type}_neural_net_model.pth')
    
    print(f"Neural network trained with {best_accuracy:.2f}% accuracy")
    return model, best_accuracy

def train_lstm(exercise_type, train_loader, val_loader, num_features):
    """Train LSTM model"""
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            out = self.fc(h_n[-1])
            return out
    
    model = LSTMClassifier(num_features, 64, 3)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))  # Add sequence dimension
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.unsqueeze(1))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'{exercise_type} - Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'models/{exercise_type}_lstm_model.pth')
    
    print(f"LSTM trained with {best_accuracy:.2f}% accuracy")
    return model, best_accuracy

def train_logistic_regression(exercise_type, X_train, y_train, X_val, y_val):
    """Train logistic regression model"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    val_accuracy = accuracy_score(y_val, model.predict(X_val_scaled))
    
    print(f'Logistic Regression - Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
    
    # Save the model
    joblib.dump(model, f'models/{exercise_type}_logistic_regression_model.pkl')
    joblib.dump(scaler, f'models/{exercise_type}_scaler.pkl')
    
    return model, val_accuracy

def train_all_models(exercise_type, data_dir):
    """Train all three models for a specific exercise"""
    try:
        dataset = ExerciseDataset(data_dir, exercise_type)
        if len(dataset) == 0:
            log_error(f"No data available for {exercise_type}")
            return None
        
        # Prepare data for all models
        all_features = []
        all_labels = []
        
        for i in range(len(dataset)):
            features, label = dataset[i]
            all_features.append(features.numpy())
            all_labels.append(label)
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create DataLoaders for neural network models
        train_set = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_set = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        
        num_features = X.shape[1]
        
        # Train all three models
        print(f"\nTraining neural network for {exercise_type}...")
        nn_model, nn_accuracy = train_neural_network(exercise_type, train_loader, val_loader, num_features)
        
        print(f"\nTraining LSTM for {exercise_type}...")
        lstm_model, lstm_accuracy = train_lstm(exercise_type, train_loader, val_loader, num_features)
        
        print(f"\nTraining logistic regression for {exercise_type}...")
        lr_model, lr_accuracy = train_logistic_regression(exercise_type, X_train, y_train, X_val, y_val)
        
        return nn_model, lstm_model, lr_model
        
    except Exception as e:
        log_error(f"Error training {exercise_type} models: {str(e)}")
        return None

def auto_train_all_exercises(data_dir):
    os.makedirs('models', exist_ok=True)
    
    available_exercises = []
    for exercise in EXERCISE_TYPES:
        exercise_dir = os.path.join(data_dir, exercise)
        if os.path.exists(exercise_dir) and len(os.listdir(exercise_dir)) > 0:
            available_exercises.append(exercise)
    
    if not available_exercises:
        log_error("No exercises with data found")
        return
    
    print(f"Found {len(available_exercises)} exercises with data: {', '.join(available_exercises)}")
    
    # Show model selection menu
    print("\nSelect model type to train:")
    print("1. Neural Network")
    print("2. LSTM")
    print("3. Logistic Regression")
    print("4. All Models")
    
    try:
        choice = int(input("Enter your choice (1-4): "))
        model_types = ["neural_net", "lstm", "logistic_regression", "all"]
        
        if 1 <= choice <= 4:
            selected_model = model_types[choice-1]
            
            # Handle "All" option differently
            if selected_model == "all":
                print(f"\nTraining ALL models for all exercises...")
                
                for exercise in available_exercises:
                    print(f"\nTraining all models for {exercise}...")
                    train_all_models(exercise, data_dir)
            else:
                print(f"\nSelected model: {selected_model.replace('_', ' ').title()}")
                
                # Train models for all available exercises
                for exercise in available_exercises:
                    print(f"\nTraining {selected_model.replace('_', ' ').title()} model for {exercise}...")
                    train_model_for_exercise(exercise, data_dir, selected_model)
        else:
            log_error("Invalid choice. Please enter a number between 1 and 4.")
            
    except ValueError:
        log_error("Invalid input. Please enter a valid number.")
    
    # Print error summary
    print("\nError Summary:")
    print(f"Total errors encountered: {ERROR_LOG['count']}")
    print(f"Incompatible data points: {ERROR_LOG['incompatible_data_points']}")
    print("Error reasons:")
    for reason, count in ERROR_LOG['reasons'].items():
        print(f"- {reason}: {count} times")

if __name__ == '__main__':
    # Default data directory (can be overridden by user input)
    data_directory = 'C:\\Users\\itsth\\Desktop\\skill issue\\ugh\\capstone\\tennis_elbow_rehab1\\data\\processed_data'
    
    # Run the training process
    auto_train_all_exercises(data_directory)