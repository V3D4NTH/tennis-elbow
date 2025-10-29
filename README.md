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
