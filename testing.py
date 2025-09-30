import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pickle
import joblib
import os
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from sklearn.preprocessing import StandardScaler

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Exercise configuration (matching your training script)
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

# Model definitions (matching your training script)
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

def safe_calculate_angle(v1, v2):
    """Safely calculate angle between two vectors"""
    try:
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
    """Extract features from landmarks sequence (matching training script)"""
    try:
        config = EXERCISE_CONFIG[exercise_type]
        features = {}
        
        if not landmarks_sequence or len(landmarks_sequence) == 0:
            return get_default_features(exercise_type)
        
        # Get the first valid frame of landmarks
        if isinstance(landmarks_sequence, list):
            if len(landmarks_sequence[0]) < 16:
                return get_default_features(exercise_type)
            landmarks = landmarks_sequence[0]
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
                features['elbow_flexion'] = 90.0
        
        elif config['primary_metric'] == 'wrist_flexion_angle':
            try:
                elbow = np.array([landmarks[13][0], landmarks[13][1], landmarks[13][2]])
                wrist = np.array([landmarks[15][0], landmarks[15][1], landmarks[15][2]])
                
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
        return get_default_features(exercise_type)

def get_default_features(exercise_type):
    """Return default features when extraction fails"""
    config = EXERCISE_CONFIG[exercise_type]
    features = {}
    
    if config['primary_metric'] == 'elbow_flexion':
        features['elbow_flexion'] = 90.0
    elif config['primary_metric'] == 'wrist_flexion_angle':
        features['wrist_flexion_angle'] = 90.0
    
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
        v2 = np.array([1, 0, 0])
        return safe_calculate_angle(v1, v2) / 180.0
    except:
        return 0.1

def calculate_movement_smoothness_fixed(landmarks_sequence):
    """Calculate movement smoothness with proper error handling"""
    try:
        if not landmarks_sequence or len(landmarks_sequence) < 2:
            return 0.5
        
        positions = []
        for landmarks in landmarks_sequence:
            if isinstance(landmarks, list) and len(landmarks) > 15:
                positions.append([landmarks[15][0], landmarks[15][1]])
        
        if len(positions) < 2:
            return 0.5
        
        positions = np.array(positions)
        velocities = np.diff(positions, axis=0)
        
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

def process_video(video_path, callback=None):
    """Process video and extract MediaPipe landmarks"""
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    )
    
    frame_count = 0
    processed_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            landmarks_list.append(landmarks)
            processed_frames += 1
        
        # Update progress if callback provided
        if callback and frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            callback(progress, f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    pose.close()
    
    return landmarks_list if landmarks_list else None

def load_model(model_path, model_type, input_size):
    """Load trained model"""
    try:
        if model_type == 'neural_net':
            model = SimpleNN(input_size)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model
        elif model_type == 'lstm':
            model = SimpleLSTM(input_size)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model
        elif model_type == 'logistic_regression':
            return joblib.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def predict_with_model(features, model, model_type, scaler=None):
    """Make prediction with the loaded model"""
    try:
        if model_type == 'logistic_regression':
            if scaler:
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
            else:
                prediction = model.predict([features])[0]
                probabilities = model.predict_proba([features])[0]
            return prediction, probabilities
        else:
            # Neural network or LSTM
            features_tensor = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                prediction = np.argmax(probabilities)
            return prediction, probabilities
    except Exception as e:
        raise Exception(f"Error making prediction: {e}")

class ExerciseEvaluatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tennis Elbow Rehabilitation Exercise Evaluator")
        self.root.geometry("1000x800")  # Increased width and height
        self.root.configure(bg='#f0f0f0')
        
        # Make window resizable
        self.root.resizable(True, True)
        
        # Variables
        self.video_path = tk.StringVar()
        self.exercise_type = tk.StringVar(value="wrist_extension")
        self.model_type = tk.StringVar(value="neural_net")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create main container with grid layout for better control
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configure grid weights for responsive layout
        main_container.grid_rowconfigure(1, weight=1)  # Results area gets extra space
        main_container.grid_columnconfigure(0, weight=1)
        
        # Top section for controls
        controls_frame = tk.Frame(main_container, bg='#f0f0f0')
        controls_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        
        # Title
        title_label = tk.Label(controls_frame, text="Tennis Elbow Rehabilitation Exercise Evaluator", 
                              font=("Arial", 16, "bold"), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 15))
        
        # Create horizontal layout for input controls
        input_frame = tk.Frame(controls_frame, bg='#f0f0f0')
        input_frame.pack(fill='x')
        
        # Left column - Video and Exercise selection
        left_column = tk.Frame(input_frame, bg='#f0f0f0')
        left_column.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        # Video Selection Frame
        video_frame = tk.LabelFrame(left_column, text="Video Selection", font=("Arial", 11, "bold"),
                                   bg='#f0f0f0', fg='#2c3e50', padx=8, pady=8)
        video_frame.pack(fill='x', pady=(0, 10))
        
        video_path_frame = tk.Frame(video_frame, bg='#f0f0f0')
        video_path_frame.pack(fill='x')
        
        tk.Label(video_path_frame, text="Selected Video:", font=("Arial", 9), 
                bg='#f0f0f0').pack(anchor='w')
        
        self.video_path_label = tk.Label(video_path_frame, textvariable=self.video_path,
                                        font=("Arial", 9), bg='white', relief='sunken',
                                        anchor='w', height=2, wraplength=400)
        self.video_path_label.pack(fill='x', pady=(2, 5))
        
        self.browse_button = tk.Button(video_path_frame, text="Browse Video File",
                                      command=self.browse_video, font=("Arial", 10),
                                      bg='#3498db', fg='white', relief='raised')
        self.browse_button.pack(pady=(0, 5))
        
        # Exercise Selection Frame
        exercise_frame = tk.LabelFrame(left_column, text="Exercise Type", font=("Arial", 11, "bold"),
                                      bg='#f0f0f0', fg='#2c3e50', padx=8, pady=8)
        exercise_frame.pack(fill='x')
        
        tk.Radiobutton(exercise_frame, text="Wrist Extension", variable=self.exercise_type,
                      value="wrist_extension", font=("Arial", 9), bg='#f0f0f0').pack(anchor='w')
        tk.Radiobutton(exercise_frame, text="Wrist Flexion", variable=self.exercise_type,
                      value="wrist_flexion", font=("Arial", 9), bg='#f0f0f0').pack(anchor='w')
        
        # Right column - Model selection and Evaluate button
        right_column = tk.Frame(input_frame, bg='#f0f0f0')
        right_column.pack(side='right', fill='x', expand=True, padx=(10, 0))
        
        # Model Selection Frame
        model_frame = tk.LabelFrame(right_column, text="Model Type", font=("Arial", 11, "bold"),
                                   bg='#f0f0f0', fg='#2c3e50', padx=8, pady=8)
        model_frame.pack(fill='x', pady=(0, 10))
        
        tk.Radiobutton(model_frame, text="Neural Network", variable=self.model_type,
                      value="neural_net", font=("Arial", 9), bg='#f0f0f0').pack(anchor='w')
        tk.Radiobutton(model_frame, text="LSTM", variable=self.model_type,
                      value="lstm", font=("Arial", 9), bg='#f0f0f0').pack(anchor='w')
        tk.Radiobutton(model_frame, text="Logistic Regression", variable=self.model_type,
                      value="logistic_regression", font=("Arial", 9), bg='#f0f0f0').pack(anchor='w')
        
        # Evaluate Button and Progress
        eval_frame = tk.Frame(right_column, bg='#f0f0f0')
        eval_frame.pack(fill='x')
        
        self.evaluate_button = tk.Button(eval_frame, text="Evaluate Exercise Form",
                                        command=self.start_evaluation, font=("Arial", 11, "bold"),
                                        bg='#27ae60', fg='white', height=2, relief='raised')
        self.evaluate_button.pack(fill='x', pady=(0, 10))
        
        # Progress Bar
        self.progress = ttk.Progressbar(eval_frame, length=300, mode='determinate')
        self.progress.pack(fill='x', pady=(0, 5))
        
        self.progress_label = tk.Label(eval_frame, text="", font=("Arial", 8), bg='#f0f0f0')
        self.progress_label.pack()
        
        # Results Frame - This gets the most space
        results_frame = tk.LabelFrame(main_container, text="Evaluation Results", 
                                     font=("Arial", 12, "bold"),
                                     bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        results_frame.grid(row=1, column=0, sticky='nsew', pady=(10, 0))
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Results Text Area with enhanced scrolling
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            height=25,  # Increased height
            width=90,   # Increased width
            font=("Consolas", 10),  # Changed to Consolas for better readability
            bg='white',
            fg='#2c3e50',
            wrap=tk.WORD,  # Word wrapping
            relief='sunken',
            borderwidth=2,
            padx=10,
            pady=10
        )
        self.results_text.grid(row=0, column=0, sticky='nsew', pady=5)
        
        # Configure scrollbar appearance
        self.results_text.configure(
            selectbackground='#3498db',
            selectforeground='white',
            insertbackground='#2c3e50'
        )
        
    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
    
    def update_progress(self, value, text=""):
        self.progress['value'] = value
        self.progress_label.config(text=text)
        self.root.update_idletasks()
    
    def start_evaluation(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file first!")
            return
        
        # Disable button during processing
        self.evaluate_button.config(state='disabled')
        self.results_text.delete(1.0, tk.END)
        
        # Start evaluation in a separate thread
        thread = threading.Thread(target=self.evaluate_video)
        thread.daemon = True
        thread.start()
    
    def evaluate_video(self):
        try:
            video_path = self.video_path.get()
            exercise_type = self.exercise_type.get()
            model_type = self.model_type.get()
            
            # Update UI
            self.update_progress(0, "Starting evaluation...")
            
            # Check if video file exists
            if not os.path.exists(video_path):
                raise Exception("Video file not found!")
            
            # Check model files
            models_dir = 'models'
            if model_type == 'logistic_regression':
                model_path = os.path.join(models_dir, f'{exercise_type}_logistic_regression_model.pkl')
                scaler_path = os.path.join(models_dir, f'{exercise_type}_scaler.pkl')
            else:
                model_path = os.path.join(models_dir, f'{exercise_type}_{model_type}_model.pth')
                scaler_path = None
            
            if not os.path.exists(model_path):
                raise Exception(f"Model file not found: {model_path}\nMake sure you have trained the models first!")
            
            # Load scaler if needed
            scaler = None
            if model_type == 'logistic_regression' and scaler_path and os.path.exists(scaler_path):
                try:
                    scaler = joblib.load(scaler_path)
                except Exception as e:
                    self.log_message(f"Warning: Could not load scaler: {e}")
            
            # Process video
            self.update_progress(10, "Processing video...")
            landmarks_sequence = process_video(video_path, self.update_progress)
            
            if landmarks_sequence is None:
                raise Exception("No pose landmarks detected in the video. Please ensure the video shows clear human poses.")
            
            # Extract features
            self.update_progress(80, "Extracting features...")
            features_dict = calculate_exercise_specific_features(landmarks_sequence, exercise_type)
            features = list(features_dict.values())
            
            # Ensure consistent feature vector length
            expected_length = len(EXERCISE_CONFIG[exercise_type]['secondary_metrics']) + 1
            if len(features) != expected_length:
                features = features[:expected_length] if len(features) > expected_length else features + [0.0] * (expected_length - len(features))
            
            # Load model
            self.update_progress(90, "Loading model...")
            model = load_model(model_path, model_type, len(features))
            
            # Make prediction
            self.update_progress(95, "Making prediction...")
            prediction, probabilities = predict_with_model(features, model, model_type, scaler)
            
            # Display results
            self.update_progress(100, "Complete!")
            self.display_results(video_path, exercise_type, model_type, prediction, 
                               probabilities, features_dict)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log_message(f"Error: {str(e)}")
        finally:
            # Re-enable button
            self.evaluate_button.config(state='normal')
            self.update_progress(0, "")
    
    def log_message(self, message):
        """Thread-safe way to add messages to results text"""
        self.root.after(0, lambda: self.results_text.insert(tk.END, message + "\n"))
    
    def display_results(self, video_path, exercise_type, model_type, prediction, probabilities, features_dict):
        class_names = ['Good Form', 'Bad Form', 'Mediocre Form']
        
        result_text = f"""
{'='*80}
EVALUATION RESULTS
{'='*80}

Exercise Type: {exercise_type.replace('_', ' ').title()}
Model Used: {model_type.replace('_', ' ').title()}
Video: {os.path.basename(video_path)}

{'='*80}
PREDICTION RESULTS
{'='*80}

Form Classification: {class_names[prediction]}
Confidence: {probabilities[prediction]:.2%}

CLASS PROBABILITIES:
"""
        
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            status = "  ✓" if i == prediction else "   "
            result_text += f"{status} {class_name:<18}: {prob:.2%}\n"
        
        result_text += f"""
{'='*80}
DETAILED FEEDBACK & RECOMMENDATIONS
{'='*80}

"""
        
        if prediction == 0:  # Good form
            result_text += """✓ EXCELLENT FORM!

Your exercise form looks great! Keep up the fantastic work.

Key strengths observed:
• Proper alignment and positioning
• Good control and stability
• Appropriate range of motion
• Smooth movement patterns

Continue with your current technique and maintain consistency in your 
rehabilitation program.

"""
        elif prediction == 1:  # Bad form
            result_text += """⚠ FORM NEEDS IMPROVEMENT

Your current form requires attention to prevent injury and maximize 
therapeutic benefits.

Areas for improvement:
• Check your posture and overall alignment
• Ensure proper elbow and wrist positioning
• Focus on slower, more controlled movements
• Pay attention to joint stability throughout the movement

RECOMMENDATIONS:
1. Review proper exercise technique with instructional materials
2. Practice in front of a mirror to monitor form
3. Consider reducing weight/resistance to focus on technique
4. Consult with a physical therapist for personalized guidance

"""
        else:  # Mediocre form
            result_text += """~ FORM IS ACCEPTABLE, BUT CAN BE IMPROVED

Your form is on the right track but has room for enhancement to 
maximize therapeutic benefits.

Areas to focus on:
• Maintain consistent alignment throughout the movement
• Ensure smoother, more controlled motion patterns
• Pay closer attention to elbow stability
• Work on achieving full range of motion safely

RECOMMENDATIONS:
1. Practice the movement slowly to build muscle memory
2. Focus on maintaining proper form over speed or repetitions
3. Consider recording yourself periodically to monitor progress
4. Gradually increase difficulty only after mastering current form

"""
        
        result_text += f"""
{'='*80}
TECHNICAL FEATURE ANALYSIS
{'='*80}

The following metrics were analyzed to evaluate your exercise form:

"""
        
        for feature_name, value in features_dict.items():
            # Format feature names nicely
            formatted_name = feature_name.replace('_', ' ').title()
            result_text += f"{formatted_name:<25}: {value:.4f}\n"
        
        result_text += f"""

{'='*80}
NEXT STEPS
{'='*80}

1. Review the feedback above and identify specific areas for improvement
2. Practice the exercise focusing on the recommended adjustments
3. Record another video after practicing to track your progress
4. Consider consulting with a healthcare provider for personalized advice

Remember: Consistency and proper form are more important than speed or 
repetition count in rehabilitation exercises.

{'='*80}
"""
        
        # Update results in main thread and auto-scroll to top
        def update_results():
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)
            self.results_text.see(tk.INSERT)  # Auto-scroll to beginning
        
        self.root.after(0, update_results)

def main():
    root = tk.Tk()
    app = ExerciseEvaluatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()