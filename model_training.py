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
from sklearn.metrics import accuracy_score, classification_report
import joblib
import math
import traceback
import warnings
warnings.filterwarnings('ignore')

# Hardcoded paths for each exercise type
WRIST_EXTENSION_DATA_PATH = r'C:\\Users\itsth\\Desktop\\skill issue\\ugh\\capstone\\testing_tennis_elbow_rehab\\data\\processed_data\\wrist_extension'
WRIST_FLEXION_DATA_PATH = r'C:\\Users\itsth\\Desktop\\skill issue\\ugh\\capstone\\testing_tennis_elbow_rehab\\data\\processed_data\\wrist_flexion'

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
EPOCHS = 50  
BATCH_SIZE = 32  
LEARNING_RATE = 0.002

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
    """Safely calculate angle between two vectors"""
    try:
        # Ensure vectors are numpy arrays with correct shape
        v1 = np.array(v1).flatten()
        v2 = np.array(v2).flatten()
        
        if len(v1) != 3 or len(v2) != 3:
            return 0.0
            
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0 and norm_v2 > 0:
            cos_theta = dot_product / (norm_v1 * norm_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return math.acos(cos_theta) * 180 / math.pi
        return 0.0
    except Exception as e:
        return 0.0

def calculate_exercise_specific_features(landmarks_sequence, exercise_type):
    """Extract features from landmarks sequence"""
    try:
        config = EXERCISE_CONFIG[exercise_type]
        features = {}
        
        # Ensure landmarks_sequence is a list and not empty
        if not landmarks_sequence or len(landmarks_sequence) == 0:
            log_error("Empty landmarks sequence")
            return get_default_features(exercise_type)
        
        # Get the first valid frame of landmarks
        if isinstance(landmarks_sequence, list):
            if len(landmarks_sequence[0]) < 16:  # Need at least 16 landmarks
                log_error(f"Insufficient landmarks: {len(landmarks_sequence[0])}")
                return get_default_features(exercise_type)
            landmarks = landmarks_sequence[0]  # Use first frame for now
        else:
            landmarks = landmarks_sequence
        
        # Primary metric calculation
        if config['primary_metric'] == 'elbow_flexion':
            try:
                shoulder = np.array([landmarks[11][0], landmarks[11][1], landmarks[11][2]])
                elbow = np.array([landmarks[13][0], landmarks[13][1], landmarks[13][2]])
                wrist = np.array([landmarks[15][0], landmarks[15][1], landmarks[15][2]])
                
                v1 = wrist - elbow
                v2 = shoulder - elbow
                features['elbow_flexion'] = safe_calculate_angle(v1, v2)
            except Exception as e:
                features['elbow_flexion'] = 90.0  # Default angle
        
        elif config['primary_metric'] == 'wrist_flexion_angle':
            try:
                elbow = np.array([landmarks[13][0], landmarks[13][1], landmarks[13][2]])
                wrist = np.array([landmarks[15][0], landmarks[15][1], landmarks[15][2]])
                
                # Calculate wrist flexion relative to vertical
                v1 = wrist - elbow
                v2 = np.array([0, -1, 0])
                features['wrist_flexion_angle'] = safe_calculate_angle(v1, v2)
            except Exception as e:
                features['wrist_flexion_angle'] = 90.0
        
        # Secondary metrics
        for metric in config['secondary_metrics']:
            if metric == 'wrist_deviation':
                features['wrist_deviation'] = calculate_wrist_deviation_fixed(landmarks)
            elif metric == 'movement_smoothness':
                features['movement_smoothness'] = calculate_movement_smoothness_fixed(landmarks_sequence)
            elif metric == 'elbow_stability':
                features['elbow_stability'] = calculate_elbow_stability_fixed(landmarks_sequence)
            elif metric == 'range_of_motion':
                features['range_of_motion'] = calculate_range_of_motion_fixed(landmarks_sequence)
        
        return features
    except Exception as e:
        log_error(f"Error in calculate_exercise_specific_features: {str(e)}")
        return get_default_features(exercise_type)

def get_default_features(exercise_type):
    """Return default features when extraction fails"""
    config = EXERCISE_CONFIG[exercise_type]
    features = {}
    
    # Set default values for primary metric
    if config['primary_metric'] == 'elbow_flexion':
        features['elbow_flexion'] = 90.0
    elif config['primary_metric'] == 'wrist_flexion_angle':
        features['wrist_flexion_angle'] = 90.0
    
    # Set default values for secondary metrics
    for metric in config['secondary_metrics']:
        if metric in ['wrist_deviation', 'elbow_stability', 'range_of_motion']:
            features[metric] = 0.1
        elif metric == 'movement_smoothness':
            features[metric] = 0.5
    
    return features

def calculate_wrist_deviation_fixed(landmarks):
    """Calculate wrist deviation with proper error handling"""
    try:
        if len(landmarks) < 16:
            return 0.1
        
        elbow = np.array([landmarks[13][0], landmarks[13][1], landmarks[13][2]])
        wrist = np.array([landmarks[15][0], landmarks[15][1], landmarks[15][2]])
        
        v1 = wrist - elbow
        v2 = np.array([1, 0, 0])  # Horizontal reference
        return safe_calculate_angle(v1, v2) / 180.0  # Normalize to [0,1]
    except:
        return 0.1

def calculate_movement_smoothness_fixed(landmarks_sequence):
    """Calculate movement smoothness with proper error handling"""
    try:
        if not landmarks_sequence or len(landmarks_sequence) < 2:
            return 0.5
        
        # Extract wrist positions from valid frames
        positions = []
        for landmarks in landmarks_sequence:
            if isinstance(landmarks, list) and len(landmarks) > 15:
                positions.append([landmarks[15][0], landmarks[15][1]])
        
        if len(positions) < 2:
            return 0.5
        
        positions = np.array(positions)
        velocities = np.diff(positions, axis=0)
        
        # Calculate smoothness as inverse of velocity variance
        smoothness = 1.0 / (1.0 + np.var(velocities))
        return float(smoothness)
    except:
        return 0.5

def calculate_elbow_stability_fixed(landmarks_sequence):
    """Calculate elbow stability with proper error handling"""
    try:
        if not landmarks_sequence:
            return 0.1
        
        elbow_positions = []
        for landmarks in landmarks_sequence:
            if isinstance(landmarks, list) and len(landmarks) > 13:
                elbow_positions.append([landmarks[13][0], landmarks[13][1], landmarks[13][2]])
        
        if len(elbow_positions) < 2:
            return 0.1
        
        elbow_positions = np.array(elbow_positions)
        stability = 1.0 / (1.0 + np.std(elbow_positions))
        return float(min(stability, 1.0))
    except:
        return 0.1

def calculate_range_of_motion_fixed(landmarks_sequence):
    """Calculate range of motion with proper error handling"""
    try:
        if not landmarks_sequence:
            return 0.1
        
        distances = []
        for landmarks in landmarks_sequence:
            if isinstance(landmarks, list) and len(landmarks) > 15:
                elbow = np.array([landmarks[13][0], landmarks[13][1], landmarks[13][2]])
                wrist = np.array([landmarks[15][0], landmarks[15][1], landmarks[15][2]])
                distances.append(np.linalg.norm(wrist - elbow))
        
        if len(distances) < 2:
            return 0.1
        
        rom = max(distances) - min(distances)
        return float(min(rom, 1.0))
    except:
        return 0.1

class ExerciseDataset(Dataset):
    def __init__(self, exercise_type):
        self.exercise_type = exercise_type
        self.data_paths = []
        self.labels = []
        self.valid_indices = []
        
        # Use hardcoded path based on exercise type
        if exercise_type == 'wrist_extension':
            base_path = WRIST_EXTENSION_DATA_PATH
        elif exercise_type == 'wrist_flexion':
            base_path = WRIST_FLEXION_DATA_PATH
        else:
            raise ValueError(f"Unsupported exercise type: {exercise_type}")
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"No data found for {exercise_type} at {base_path}")
        
        # Load all data paths and labels
        for subfolder in os.listdir(base_path):
            subfolder_path = os.path.join(base_path, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.endswith('.pkl'):
                        filepath = os.path.join(subfolder_path, filename)
                        self.data_paths.append(filepath)
                        
                        # Extract label from filename - FIXED LOGIC
                        filename_lower = filename.lower()
                        if 'bad' in filename_lower:
                            self.labels.append(1)  # Bad form
                        elif 'mediocre' in filename_lower:
                            self.labels.append(2)  # Mediocre form
                        else:
                            # Default to good form if neither 'bad' nor 'mediocre' is found
                            self.labels.append(0)  # Good form
        
        # Pre-validate data
        print(f"Loading and validating {len(self.data_paths)} samples for {exercise_type}...")
        for idx in range(len(self.data_paths)):
            try:
                with open(self.data_paths[idx], 'rb') as f:
                    data = pickle.load(f)
                    if data and len(data) > 0:
                        self.valid_indices.append(idx)
            except:
                continue
        
        print(f"Found {len(self.valid_indices)} valid samples out of {len(self.data_paths)}")
        
        # Print label distribution for verification
        valid_labels = [self.labels[i] for i in self.valid_indices]
        unique, counts = np.unique(valid_labels, return_counts=True)
        print(f"Label distribution after loading:")
        for label, count in zip(unique, counts):
            label_name = ['Good', 'Bad', 'Mediocre'][label]
            print(f"  {label_name}: {count} samples")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        try:
            with open(self.data_paths[actual_idx], 'rb') as f:
                landmarks_sequence = pickle.load(f)
            
            features = calculate_exercise_specific_features(landmarks_sequence, self.exercise_type)
            feature_vector = list(features.values())
            
            # Ensure consistent feature vector length
            expected_length = len(EXERCISE_CONFIG[self.exercise_type]['secondary_metrics']) + 1
            if len(feature_vector) != expected_length:
                feature_vector = feature_vector[:expected_length] if len(feature_vector) > expected_length else feature_vector + [0.0] * (expected_length - len(feature_vector))
            
            return torch.tensor(feature_vector, dtype=torch.float32), self.labels[actual_idx]
        except Exception as e:
            # Return default features on error
            features = get_default_features(self.exercise_type)
            feature_vector = list(features.values())
            return torch.tensor(feature_vector, dtype=torch.float32), self.labels[actual_idx]

class SimpleNN(nn.Module):
    """Simple neural network with better architecture"""
    def __init__(self, input_size, hidden_size=32, num_classes=3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class SimpleLSTM(nn.Module):
    """Fixed LSTM model"""
    def __init__(self, input_size, hidden_size=32, num_layers=2, num_classes=3):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Initialize hidden states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Use last output
        out = self.fc(out[:, -1, :])
        return out

def train_neural_network(exercise_type, train_loader, val_loader, num_features):
    """Train neural network with proper early stopping"""
    model = SimpleNN(num_features)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_accuracy = 0
    best_model_state = None
    patience = 15
    counter = 0
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        print(f'{exercise_type} NN - Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            counter = 0
            print(f"New best model with validation accuracy: {val_accuracy:.2f}%")
        else:
            counter += 1
            # Don't allow early stopping if accuracy is below 60%
            if counter >= patience and val_accuracy >= 60.0:
                print(f"Early stopping at epoch {epoch+1}")
                break
            elif counter >= patience and val_accuracy < 60.0:
                print(f"Accuracy {val_accuracy:.2f}% is below 60%, continuing training...")
                counter = 0  # Reset counter to continue training
    
    # Save best model
    if best_model_state is not None:
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{exercise_type}_neural_net_model.pth'
        torch.save(best_model_state, model_path)
        print(f"Saved best model with accuracy: {best_accuracy:.2f}%")
    
    return model, best_accuracy

def train_lstm(exercise_type, train_loader, val_loader, num_features):
    """Train LSTM with fixed dimensions"""
    model = SimpleLSTM(num_features)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_accuracy = 0
    best_model_state = None
    patience = 15
    counter = 0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        print(f'{exercise_type} LSTM - Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            counter = 0
            print(f"New best model with validation accuracy: {val_accuracy:.2f}%")
        else:
            counter += 1
            # Don't allow early stopping if accuracy is below 60%
            if counter >= patience and val_accuracy >= 60.0:
                print(f"Early stopping at epoch {epoch+1}")
                break
            elif counter >= patience and val_accuracy < 60.0:
                print(f"Accuracy {val_accuracy:.2f}% is below 60%, continuing training...")
                counter = 0  # Reset counter to continue training
    
    # Save best model
    if best_model_state is not None:
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{exercise_type}_lstm_model.pth'
        torch.save(best_model_state, model_path)
        print(f"Saved best LSTM model with accuracy: {best_accuracy:.2f}%")
    
    return model, best_accuracy

def train_logistic_regression(exercise_type, X_train, y_train, X_val, y_val):
    """Train logistic regression with proper scaling"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Try different C values for regularization
    best_model = None
    best_accuracy = 0
    best_c = None
    
    for c_value in [0.001, 0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(max_iter=1000, C=c_value, penalty='l2', solver='lbfgs')
        model.fit(X_train_scaled, y_train)
        
        val_predictions = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, val_predictions) * 100
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_c = c_value
    
    train_accuracy = accuracy_score(y_train, best_model.predict(X_train_scaled)) * 100
    
    print(f'{exercise_type} Logistic Regression (C={best_c}) - '
          f'Train Acc: {train_accuracy:.2f}%, Val Acc: {best_accuracy:.2f}%')
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{exercise_type}_logistic_regression_model.pkl'
    scaler_path = f'models/{exercise_type}_scaler.pkl'
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved Logistic Regression model with accuracy: {best_accuracy:.2f}%")
    
    # Print classification report
    val_predictions = best_model.predict(X_val_scaled)
    print("\nClassification Report:")
    print(classification_report(y_val, val_predictions, 
                              target_names=['Good', 'Bad', 'Mediocre']))
    
    return best_model, best_accuracy

def train_all_models(exercise_type):
    """Train all three models for a specific exercise"""
    try:
        print(f"\n{'='*50}")
        print(f"Training models for {exercise_type}")
        print(f"{'='*50}")
        
        dataset = ExerciseDataset(exercise_type)
        if len(dataset) == 0:
            print(f"No valid data available for {exercise_type}")
            return None
        
        # Prepare data
        all_features = []
        all_labels = []
        
        for i in range(len(dataset)):
            features, label = dataset[i]
            all_features.append(features.numpy())
            all_labels.append(label)
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution:")
        for label, count in zip(unique, counts):
            label_name = ['Good', 'Bad', 'Mediocre'][label]
            print(f"  {label_name}: {count} samples")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        
        # Create DataLoaders
        train_set = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long)
        )
        val_set = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32), 
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
        
        num_features = X.shape[1]
        print(f"Number of features: {num_features}")
        
        # Train all models
        results = {}
        
        print(f"\n{'-'*40}")
        print("Training Neural Network...")
        print(f"{'-'*40}")
        nn_model, nn_accuracy = train_neural_network(exercise_type, train_loader, val_loader, num_features)
        results['Neural Network'] = nn_accuracy
        
        print(f"\n{'-'*40}")
        print("Training LSTM...")
        print(f"{'-'*40}")
        lstm_model, lstm_accuracy = train_lstm(exercise_type, train_loader, val_loader, num_features)
        results['LSTM'] = lstm_accuracy
        
        print(f"\n{'-'*40}")
        print("Training Logistic Regression...")
        print(f"{'-'*40}")
        lr_model, lr_accuracy = train_logistic_regression(exercise_type, X_train, y_train, X_val, y_val)
        results['Logistic Regression'] = lr_accuracy
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Results Summary for {exercise_type}:")
        print(f"{'='*50}")
        for model_name, accuracy in results.items():
            print(f"{model_name}: {accuracy:.2f}%")
        
        best_model = max(results, key=results.get)
        print(f"\nBest Model: {best_model} ({results[best_model]:.2f}%)")
        
        return results
        
    except Exception as e:
        print(f"Error training {exercise_type} models: {str(e)}")
        traceback.print_exc()
        return None

def auto_train_all_exercises():
    """Main training function"""
    print("\nStarting Model Training Pipeline")
    print("="*60)
    
    # Show model selection menu
    print("\nSelect training mode:")
    print("1. Neural Network only")
    print("2. LSTM only")
    print("3. Logistic Regression only")
    print("4. All Models (Recommended)")
    
    try:
        choice = int(input("\nEnter your choice (1-4): "))
        
        if choice == 4:
            # Train all models for all exercises
            all_results = {}
            
            for exercise in EXERCISE_TYPES:
                results = train_all_models(exercise)
                if results:
                    all_results[exercise] = results
            
            # Print final summary
            print("\n" + "="*60)
            print("FINAL TRAINING SUMMARY")
            print("="*60)
            
            for exercise, results in all_results.items():
                print(f"\n{exercise.replace('_', ' ').title()}:")
                for model, acc in results.items():
                    print(f"  {model}: {acc:.2f}%")
        else:
            # Train specific model type
            model_types = {1: "neural_net", 2: "lstm", 3: "logistic_regression"}
            selected_model = model_types.get(choice)
            
            if selected_model:
                print(f"\nTraining {selected_model.replace('_', ' ').title()} for all exercises...")
                
                for exercise in EXERCISE_TYPES:
                    print(f"\n{'='*50}")
                    print(f"Training {selected_model} for {exercise}")
                    print(f"{'='*50}")
                    
                    dataset = ExerciseDataset(exercise)
                    if len(dataset) == 0:
                        print(f"No valid data for {exercise}")
                        continue
                    
                    # Prepare data
                    all_features = []
                    all_labels = []
                    
                    for i in range(len(dataset)):
                        features, label = dataset[i]
                        all_features.append(features.numpy())
                        all_labels.append(label)
                    
                    X = np.array(all_features)
                    y = np.array(all_labels)
                    
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    if selected_model == "neural_net":
                        train_set = torch.utils.data.TensorDataset(
                            torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.long)
                        )
                        val_set = torch.utils.data.TensorDataset(
                            torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long)
                        )
                        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
                        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
                        train_neural_network(exercise, train_loader, val_loader, X.shape[1])
                        
                    elif selected_model == "lstm":
                        train_set = torch.utils.data.TensorDataset(
                            torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.long)
                        )
                        val_set = torch.utils.data.TensorDataset(
                            torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long)
                        )
                        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
                        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
                        train_lstm(exercise, train_loader, val_loader, X.shape[1])
                        
                    elif selected_model == "logistic_regression":
                        train_logistic_regression(exercise, X_train, y_train, X_val, y_val)
            else:
                print("Invalid choice!")
                
    except ValueError:
        print("Invalid input. Please enter a number between 1 and 4.")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
    
    # Print error summary
    print("\n" + "="*60)
    print("ERROR SUMMARY")
    print("="*60)
    print(f"Total errors: {ERROR_LOG['count']}")
    if ERROR_LOG['reasons']:
        print("\nError breakdown:")
        for reason, count in sorted(ERROR_LOG['reasons'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {reason}: {count} times")

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run the training pipeline
    auto_train_all_exercises()