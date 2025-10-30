# Tennis Elbow Rehab

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure
```
tennis_elbow_rehab/
├── data/
│   ├── raw_videos/               # Raw video files organized by exercise
│   │   ├── wrist_extension/
│   │   │   ├── Wrist_Extension_Strengthening_1/
│   │   │   ├── Wrist_Extension_Strengthening_2/
│   │   │   ├── Wrist_Extension_Strengthening_3/
│   │   │   └── Wrist_Extension_Stretch/
│   │   └── wrist_flexion/
│   │       ├── Wrist_Flexion_Strengthening_1/
│   │       ├── Wrist_Flexion_Strengthening_2/
│   │       ├── Wrist_Flexion_Strengthening_3/
│   │       └── Wrist_Flexion_Stretch/
│   └── processed_data/           # Pickled landmarks/features
├── models/
│   ├── wrist_extension_model.pth
│   └── wrist_flexion_model.pth
├── src/
│   ├── data_processing.py        # Video processing and landmark extraction
│   ├── feature_engineering.py    # Biomechanical feature computation
│   ├── model_training.py         # Model training scripts
│   ├── real_time_eval.py         # Real-time evaluation
│   └── ui_streamlit.py           # Streamlit UI
├── requirements.txt
└── README.md
```

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Process Videos
```bash
python src/data_processing.py
```

### 3. Train Models
```bash
python src/model_training.py
```

### 4. Real-time Evaluation
```bash
python src/real_time_eval.py
```

### 5. Launch Web Interface
```bash
streamlit run src/ui_streamlit.py
```

## Important Notes
- Always run commands from the project root directory (tennis-elbow/)
- Ensure you have processed data before training models
- Models will be saved in the `models/` directory
- The system supports both CPU and GPU training automatically

# How It Works (So Far)

# Tennis Elbow CNN Model Architecture

## Architecture Overview

**Model Type:** 1D Temporal Convolutional Neural Network (CNN)

**Purpose:** Classifies exercise form quality (Good/Bad/Mediocre) from pose landmark sequences

---

## Network Architecture

### Input Layer
- **Shape:** `(batch_size, channels, sequence_length)`
- **Channels:** 99 (33 MediaPose landmarks × 3 coordinates: x, y, z) OR 66 (for 2D: x, y only)
- **Sequence Length:** 60 frames (fixed, padded/truncated)

### Layer Breakdown

**Block 1:**
- Conv1D: 99→64 channels, kernel=3, padding=1
- BatchNorm1D: 64 channels
- ReLU activation
- MaxPool1D: kernel=2 (reduces sequence length by half)

**Block 2:**
- Conv1D: 64→128 channels, kernel=3, padding=1
- BatchNorm1D: 128 channels
- ReLU activation
- MaxPool1D: kernel=2

**Block 3:**
- Conv1D: 128→256 channels, kernel=3, padding=1
- BatchNorm1D: 256 channels
- ReLU activation
- MaxPool1D: kernel=2

**Global Pooling:**
- AdaptiveAvgPool1D: Reduces to single value per channel

**Fully Connected Layers:**
- FC1: 256→128 neurons + ReLU + Dropout(0.5)
- FC2: 128→3 neurons (output: 3 classes)

### Total Layers: 11
- 3 Conv1D layers
- 3 BatchNorm layers
- 3 MaxPool layers
- 2 Fully Connected layers

---

## Data Pipeline

### 1. Data Processing (`data_processing.py`)
- Extracts pose landmarks from videos using MediaPipe
- Saves as `.pkl` files with shape: `(num_frames, 33, 3)`

### 2. Dataset Loading (`ExerciseDataset`)
- Loads `.pkl` files
- **Handles multiple formats:**
  - 3D: `(seq_len, 33, 3)` → flattened to `(99, seq_len)`
  - 2D: `(seq_len, 33, 2)` → flattened to `(66, seq_len)`
  - Already flattened: just transpose
- **Pads/Truncates** all sequences to exactly 60 frames
- **Labels:** Extracted from filename (good/bad/mediocre)

### 3. Training (`train_model_for_exercise`)
- 80/20 train/validation split
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Epochs: 20
- **Auto-detects GPU/CPU** and uses available device
- **Saves to CPU** for portability across machines

---

## Key Features

### Adaptive Input Handling
✅ Auto-detects 2D (66 channels) vs 3D (99 channels) landmarks  
✅ Handles variable sequence lengths (pads/truncates to 60)  
✅ Works with different data formats automatically

### GPU/CPU Support
✅ Auto-detects CUDA availability  
✅ Trains on GPU if available, CPU otherwise  
✅ Models saved on CPU for cross-platform compatibility  
✅ Real-time eval loads models to correct device

### Robust Error Handling
✅ Checks if models exist before loading  
✅ Tries multiple channel configurations (99→66)  
✅ Graceful degradation with helpful error messages  
✅ Handles missing data files

---

## How It Works End-to-End

### 1. Video → Landmarks
- MediaPipe extracts 33 pose keypoints per frame
- Each keypoint has (x, y, z) coordinates
- Saved as pickle files

### 2. Landmarks → Features
- CNN automatically learns features from raw landmarks
- No hand-crafted feature engineering needed
- Temporal patterns captured through 1D convolutions

### 3. Training
- Loads sequences, pads to 60 frames
- Reshapes to (99, 60) or (66, 60)
- Passes through CNN
- Outputs 3 class probabilities
- Optimizes with backpropagation

### 4. Real-time Evaluation
- Captures webcam frames
- Extracts landmarks with MediaPipe
- Buffers last 60 frames
- Preprocesses same as training
- CNN predicts form quality
- Displays result on screen

---

## Why This Architecture?

### 1D Conv instead of 2D/3D
- Pose data is **temporal sequence** (not spatial image)
- 1D Conv perfect for time-series landmark data
- Faster and more efficient than 2D/3D Conv

### 3 Conv Blocks
- Progressively learns hierarchical features:
  - **Block 1:** Basic joint movements
  - **Block 2:** Multi-joint coordination
  - **Block 3:** Complex exercise patterns

### Global Average Pooling
- Makes network robust to slight sequence length variations
- Reduces parameters, prevents overfitting

### Dropout
- 50% dropout before final layer
- Prevents overfitting on small datasets

---

## Model Statistics

- **Parameters:** ~few hundred thousand (depends on input channels)
- **Input Size:** 99 × 60 = 5,940 values per sample
- **Output:** 3 class probabilities (softmax)
- **Training Time:** ~few minutes for 20 epochs (GPU)
- **Inference Speed:** Real-time (30+ FPS)

---

## Supported Exercises

1. **Wrist Extension**
2. **Wrist Flexion**

Each has its own trained model saved as `.pth` file.
