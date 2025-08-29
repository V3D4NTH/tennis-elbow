import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EXERCISE_TYPES = ['wrist_extension', 'wrist_flexion']

class ExerciseDataset(Dataset):
    def __init__(self, data_dir, exercise_type):
        self.exercise_type = exercise_type
        self.data_paths = []
        self.labels = []
        
        exercise_dir = os.path.join(data_dir, exercise_type)
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
        with open(self.data_paths[idx], 'rb') as f:
            landmarks_sequence = pickle.load(f)
        
        features = calculate_exercise_specific_features(landmarks_sequence, self.exercise_type)
        feature_vector = list(features.values())
        
        return torch.tensor(feature_vector, dtype=torch.float32), self.labels[idx]

def train_exercise_models(data_dir):
    """Train models for both exercise types"""
    for exercise_type in EXERCISE_TYPES:
        print(f"\nTraining model for {exercise_type}...")
        
        # Prepare dataset
        dataset = ExerciseDataset(data_dir, exercise_type)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        
        # Initialize model
        model = STGCN(
            num_nodes=33,
            num_features=len(EXERCISE_CONFIG[exercise_type]['secondary_metrics']) + 1,
            num_classes=3
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        best_accuracy = 0
        for epoch in range(50):
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
            print(f'Epoch [{epoch+1}/50], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), f'models/{exercise_type}_model.pth')