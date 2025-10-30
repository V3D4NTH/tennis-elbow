import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import math
import traceback

# Detect available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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



class TemporalCNN(nn.Module):
    """1D CNN for temporal landmark sequences - 11 Total Layers"""
    def __init__(self, in_channels=99, num_classes=3):
        super(TemporalCNN, self).__init__()
        
        # in_channels can be 66 (2D landmarks: x,y) or 99 (3D landmarks: x,y,z)
        
        # LAYER 1: First Conv1D block
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        # LAYER 2: BatchNorm
        self.bn1 = nn.BatchNorm1d(64)
        # LAYER 3: MaxPool (reduces sequence length by half)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # LAYER 4: Second Conv1D block
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # LAYER 5: BatchNorm
        self.bn2 = nn.BatchNorm1d(128)
        # LAYER 6: MaxPool
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # LAYER 7: Third Conv1D block
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # LAYER 8: BatchNorm
        self.bn3 = nn.BatchNorm1d(256)
        # LAYER 9: MaxPool
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Global average pooling (not counted as layer - no learnable parameters)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # LAYER 10: First Fully Connected layer
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)  # Regularization, not a layer
        # LAYER 11: Output Fully Connected layer
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()  # Activation function, not a separate layer
    
    

    def forward(self, x):
        # x shape: (batch, num_landmarks*num_coords, sequence_length)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

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
            
            # Convert to numpy array
            landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
            
            # Handle different input shapes
            if len(landmarks_array.shape) == 2:
                # Already flattened: (seq_len, num_features)
                seq_len, num_features = landmarks_array.shape
                
                # Determine if it's 2D or 3D coords
                if num_features == 66:  # 33 landmarks * 2 coords (x, y)
                    num_landmarks = 33
                    num_coords = 2
                elif num_features == 99:  # 33 landmarks * 3 coords (x, y, z)
                    num_landmarks = 33
                    num_coords = 3
                else:
                    # Unknown format, try to infer
                    num_landmarks = num_features // 3 if num_features % 3 == 0 else num_features // 2
                    num_coords = 3 if num_features % 3 == 0 else 2
                
                # Already in correct format, just transpose
                landmarks_flat = landmarks_array.T  # Shape: (num_features, seq_len)
                
            elif len(landmarks_array.shape) == 3:
                # Shape: (seq_len, num_landmarks, num_coords)
                seq_len, num_landmarks, num_coords = landmarks_array.shape
                
                # Flatten and transpose
                landmarks_flat = landmarks_array.reshape(seq_len, num_landmarks * num_coords).T
            
            else:
                raise ValueError(f"Unexpected data shape: {landmarks_array.shape}")
            
            # Pad or truncate to fixed length
            target_length = 60  # Fixed sequence length
            current_length = landmarks_flat.shape[1]
            
            if current_length < target_length:
                # Pad with zeros
                padding = np.zeros((landmarks_flat.shape[0], target_length - current_length), dtype=np.float32)
                landmarks_flat = np.concatenate([landmarks_flat, padding], axis=1)
            elif current_length > target_length:
                # Truncate
                landmarks_flat = landmarks_flat[:, :target_length]
            
            # Determine input channels for return
            num_channels = landmarks_flat.shape[0]
            
            return torch.tensor(landmarks_flat, dtype=torch.float32), self.labels[idx]
            
        except Exception as e:
            print(f"Error loading data for item {idx}: {e}")
            traceback.print_exc()
            # Return dummy data with standard shape (99 channels for 33 landmarks * 3 coords)
            return torch.zeros((99, 60), dtype=torch.float32), 0

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
        
        # In train_model_for_exercise, replace model initialization with:
        # Get first sample to determine input channels
        sample_data, _ = dataset[0]
        in_channels = sample_data.shape[0]  # Will be 66 or 99

        model = TemporalCNN(in_channels=in_channels, num_classes=3)
        model = model.to(device)  # Move model to device
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        best_accuracy = 0
        for epoch in range(20):  # Reduced epochs for quicker testing
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
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
                    inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'{exercise_type} - Epoch [{epoch+1}/20], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Save model state dict (always save on CPU for portability)
                torch.save(model.cpu().state_dict(), f'models/{exercise_type}_model.pth')
                model = model.to(device)  # Move back to device for continued training
        
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
    auto_train_all_exercises('data/processed_data')
