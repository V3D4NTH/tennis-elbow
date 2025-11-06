#!/usr/bin/env python3
"""
LSTM-based Wrist Exercise Quality Classifier
Based on: "An Efficient Patient Activity Recognition using LSTM Network 
and High-Fidelity Body Pose Tracking" (IJACSA, 2022)

Architecture:
- 4-layer LSTM with 50 units each
- Dropout layers for regularization
- Time-steps K=20 (optimal from paper)
- Processes MediaPipe 33 landmark skeleton data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------  defaults  -------------------------------------------------------
DEFAULT_DATA_ROOT  = Path(r"C:\Users\itsth\Downloads\wrist_extension_processed-20251105T093718Z-1-001")
DEFAULT_EPOCHS     = 20  # Paper used 50 epochs
DEFAULT_BATCH      = 32
DEFAULT_LR         = 1e-3
DEFAULT_DROPOUT    = 0.5
DEFAULT_TIME_STEPS = 20  # Paper found K=20 optimal (96.84% vs 96.44% at K=15)
DEFAULT_OUTPUT_DIR = Path("models")

# Exercise configurations
EXERCISES = ["wrist_extension", "wrist_flexion"]

# Class mapping (consistent with ST-GCN)
CLASS_NAMES = {0: "good", 1: "bad", 2: "mediocre"}
NUM_CLASSES = 3

# MediaPipe provides 33 landmarks, each with (x, y, z, visibility) = 132 features
NUM_LANDMARKS = 33
FEATURES_PER_LANDMARK = 4  # x, y, z, visibility
TOTAL_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 132


# ----------  Multi-Layer LSTM Model (Paper Architecture)  --------------------
class PatientActivityLSTM(nn.Module):
    """
    4-layer LSTM network as described in the reference paper.
    
    Architecture from paper:
    - 4 LSTM layers with 50 units each
    - Dropout after each LSTM layer
    - Dense layer for feature interpretation
    - Softmax output layer
    """
    def __init__(self, input_size=132, hidden_size=20, num_layers=4, 
                 num_classes=3, dropout=0.5):
        super(PatientActivityLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-layer LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # Dropout between LSTM layers
            bidirectional=False
        )
        
        # Additional dropout after LSTM (paper mentions dropout to reduce overfitting)
        self.dropout = nn.Dropout(dropout)
        
        # Dense fully connected layer (paper: "dense fully connected layer with 27 units")
        # We use 64 units for interpretation before final classification
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Final output layer with softmax (paper uses softmax for multi-class)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, time_steps, features)
            lengths: actual sequence lengths before padding
        """
        # LSTM forward pass
        # lstm_out: (batch, time_steps, hidden_size)
        # h_n: (num_layers, batch, hidden_size) - final hidden state
        # c_n: (num_layers, batch, hidden_size) - final cell state
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last relevant output for each sequence
        if lengths is not None:
            # Gather the output at the actual last time step for each sequence
            batch_size = x.size(0)
            idx = (lengths - 1).view(-1, 1, 1).expand(batch_size, 1, self.hidden_size)
            last_output = lstm_out.gather(1, idx).squeeze(1)
        else:
            # Take the last time step output
            last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Dense layer for interpretation
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Final classification layer (softmax applied in loss function)
        out = self.fc2(out)
        
        return out


# ----------  Dataset with Time-Steps Windowing  ------------------------------
class SkeletonSequenceDataset(Dataset):
    """
    Loads skeleton sequences from pickle files and creates time-step windows.
    
    From paper: "The time-steps K are the most critical parameters affecting 
    model performance. The time-steps are how many lagged variables the model 
    receives as input to forecast the next step."
    """
    def __init__(self, exercise: str, data_root: Path, time_steps: int = 20):
        self.root = data_root / exercise
        self.time_steps = time_steps
        self.pkl_files = list(self.root.rglob("*.pkl"))
        
        if not self.pkl_files:
            raise FileNotFoundError(f"No pickles in {self.root}")
        
        print(f"  Found {len(self.pkl_files)} pickle files for {exercise}")
    
    def __len__(self):
        return len(self.pkl_files)
    
    def _label_from_name(self, name: str) -> int:
        """Extract label from filename (good=0, bad=1, mediocre=2)"""
        nm = name.lower()
        return 1 if "bad" in nm else 2 if "mediocre" in nm else 0
    
    def _extract_features(self, frame):
        """
        Extract features from a single frame.
        MediaPipe provides 33 landmarks, each with (x, y, z, visibility).
        
        Args:
            frame: List of 33 landmarks, each landmark is (x, y, z) or (x, y, z, visibility)
        
        Returns:
            features: numpy array of shape (132,) = 33 landmarks × 4 features
        """
        features = []
        for landmark in frame:
            if len(landmark) == 3:  # (x, y, z)
                features.extend([landmark[0], landmark[1], landmark[2], 1.0])  # visibility=1.0
            elif len(landmark) >= 4:  # (x, y, z, visibility, ...)
                features.extend(landmark[:4])
            else:
                features.extend([0, 0, 0, 0])  # Missing landmark
        
        return np.array(features, dtype=np.float32)
    
    def __getitem__(self, idx):
        pkl = self.pkl_files[idx]
        
        with open(pkl, 'rb') as fh:
            sequence = pickle.load(fh)
        
        if not sequence:  # Empty sequence
            return None, None, None
        
        label = self._label_from_name(pkl.stem)
        
        # Extract features from all frames
        features_sequence = []
        for frame in sequence:
            frame_features = self._extract_features(frame)
            features_sequence.append(frame_features)
        
        features_sequence = np.array(features_sequence, dtype=np.float32)
        
        # Sequence length
        seq_length = len(features_sequence)
        
        # Convert to tensor
        tensor = torch.from_numpy(features_sequence)
        
        return tensor, seq_length, label


def create_time_steps_sequences(sequences, lengths, labels, time_steps=20):
    """
    Create fixed-length time-step windows from variable-length sequences.
    
    From paper: Different time-steps (5, 10, 15, 20) were tested, with 20 
    providing the best performance (96.84% accuracy).
    
    Args:
        sequences: List of tensors with variable lengths
        lengths: List of sequence lengths
        labels: List of labels
        time_steps: Number of time steps per window (K)
    
    Returns:
        windowed_sequences: Tensor of shape (N, time_steps, features)
        windowed_labels: Tensor of shape (N,)
    """
    windowed_sequences = []
    windowed_labels = []
    
    for seq, length, label in zip(sequences, lengths, labels):
        # seq shape: (length, features)
        if length >= time_steps:
            # Create sliding windows
            for i in range(length - time_steps + 1):
                window = seq[i:i+time_steps]
                windowed_sequences.append(window)
                windowed_labels.append(label)
        else:
            # Pad sequence if too short
            padding = torch.zeros((time_steps - length, seq.shape[1]), dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=0)
            windowed_sequences.append(padded_seq)
            windowed_labels.append(label)
    
    windowed_sequences = torch.stack(windowed_sequences)
    windowed_labels = torch.tensor(windowed_labels, dtype=torch.long)
    
    return windowed_sequences, windowed_labels


# ----------  Training & Evaluation  -------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(val_accs, label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names.values(),
                yticklabels=class_names.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_model_for_exercise(exercise, args, device):
    """Train LSTM model for a specific exercise"""
    print(f"\n{'='*70}")
    print(f"Training LSTM for {exercise.upper()}")
    print(f"{'='*70}")
    
    # Create output directory
    model_dir = args.output_dir / exercise
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_path = model_dir / f"lstm_{exercise}_model.pth"
    history_plot_path = model_dir / f"lstm_{exercise}_history.png"
    cm_plot_path = model_dir / f"lstm_{exercise}_confusion_matrix.png"
    
    # Load dataset
    try:
        dataset = SkeletonSequenceDataset(exercise, args.data_root, args.time_steps)
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")
        print(f"  Skipping {exercise} training")
        return None
    
    # Collect all sequences and labels
    all_sequences = []
    all_lengths = []
    all_labels = []
    
    print("  Loading sequences...")
    for i in range(len(dataset)):
        seq, length, label = dataset[i]
        if seq is not None:
            all_sequences.append(seq)
            all_lengths.append(length)
            all_labels.append(label)
    
    if len(all_sequences) == 0:
        print(f"⚠ Warning: No valid sequences found for {exercise}")
        return None
    
    print(f"  Loaded {len(all_sequences)} sequences")
    
    # Create time-step windows
    print(f"  Creating time-step windows (K={args.time_steps})...")
    windowed_sequences, windowed_labels = create_time_steps_sequences(
        all_sequences, all_lengths, all_labels, args.time_steps
    )
    
    print(f"  Total windows created: {len(windowed_sequences)}")
    print(f"  Window shape: {windowed_sequences.shape}")
    
    # Class distribution
    unique_labels, counts = torch.unique(windowed_labels, return_counts=True)
    print(f"  Class distribution in windows:")
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        print(f"    {CLASS_NAMES[label]}: {count} windows ({count/len(windowed_labels)*100:.1f}%)")
    
    # Split data (80-20 as per paper)
    train_idx, val_idx = train_test_split(
        range(len(windowed_sequences)),
        test_size=0.2,
        stratify=windowed_labels.numpy(),
        random_state=42
    )
    
    train_sequences = windowed_sequences[train_idx]
    train_labels = windowed_labels[train_idx]
    val_sequences = windowed_sequences[val_idx]
    val_labels = windowed_labels[val_idx]
    
    print(f"  Training windows: {len(train_sequences)}")
    print(f"  Validation windows: {len(val_sequences)}")
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_sequences, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_sequences, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    # Initialize model (paper architecture)
    model = PatientActivityLSTM(
        input_size=TOTAL_FEATURES,  # 132 features
        hidden_size=20,  # Paper uses 50 units per LSTM layer
        num_layers=4,    # Paper uses 4 LSTM layers
        num_classes=NUM_CLASSES,
        dropout=args.dropout
    ).to(device)
    
    print(f"\n  Model Architecture:")
    print(f"    Input size: {TOTAL_FEATURES} features")
    print(f"    LSTM layers: 4 layers × 20 units")
    print(f"    Dropout: {args.dropout}")
    print(f"    Output classes: {NUM_CLASSES}")
    
    # Loss and optimizer (paper uses Adam and categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler (paper uses ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, factor=0.5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f"\n  Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_true = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{args.epochs}] | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': CLASS_NAMES,
                'time_steps': args.time_steps,
            }, model_save_path)
            print(f"  → New best model saved! Val Acc: {val_acc*100:.2f}%")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n  ✓ Training Complete")
    print(f"    Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"    Training time: {training_time/60:.2f} minutes")
    print(f"    Model saved to: {model_save_path}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation metrics
    _, final_acc, final_preds, final_true = evaluate(model, val_loader, criterion, device)
    
    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(final_true, final_preds, 
                                target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
                                digits=4))
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, history_plot_path)
    print(f"  Training history saved to: {history_plot_path}")
    
    # Plot confusion matrix
    plot_confusion_matrix(final_true, final_preds, CLASS_NAMES, cm_plot_path)
    print(f"  Confusion matrix saved to: {cm_plot_path}")
    
    return {
        'exercise': exercise,
        'best_val_acc': best_val_acc,
        'final_val_acc': final_acc,
        'model_path': model_save_path,
        'training_time': training_time
    }


# ----------  Main  ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM models for wrist exercise quality classification"
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Processed data folder")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Training epochs (paper used 50)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help="Batch size (paper used 32)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT,
                        help="Dropout rate (paper used 0.5)")
    parser.add_argument("--time_steps", type=int, default=DEFAULT_TIME_STEPS,
                        help="Time steps K (paper found 20 optimal)")
    parser.add_argument("--exercises", nargs='+', default=EXERCISES,
                        help="Exercises to train")
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("LSTM TRAINING FOR WRIST EXERCISE QUALITY CLASSIFICATION")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Exercises: {args.exercises}")
    print(f"Time steps (K): {args.time_steps}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Dropout: {args.dropout}")
    
    # Train models
    results = []
    total_start = time.time()
    
    for exercise in args.exercises:
        result = train_model_for_exercise(exercise, args, device)
        if result:
            results.append(result)
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"\nModels trained:")
    for result in results:
        print(f"  • {result['exercise']}:")
        print(f"      Best val accuracy: {result['best_val_acc']*100:.2f}%")
        print(f"      Final val accuracy: {result['final_val_acc']*100:.2f}%")
        print(f"      Training time: {result['training_time']/60:.2f} min")
        print(f"      → {result['model_path']}")
    
    if len(results) < len(args.exercises):
        print(f"\n⚠ Warning: {len(args.exercises) - len(results)} model(s) could not be trained")
    
    print(f"\n{'='*70}")
    print("Reference: 'An Efficient Patient Activity Recognition using LSTM")
    print("Network and High-Fidelity Body Pose Tracking' (IJACSA, 2022)")
    print("Paper achieved 96.84% accuracy with K=20 time-steps")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
