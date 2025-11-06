#!/usr/bin/env python3
"""
LSTM-based Wrist Exercise Quality Testing/Inference
Loads trained LSTM models and performs real-time classification
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import cv2
import mediapipe as mp
import argparse
from pathlib import Path
from collections import deque
import time

# Import model architecture from training script
import sys
sys.path.append(str(Path(__file__).parent))
from train_lstm import PatientActivityLSTM, CLASS_NAMES, TOTAL_FEATURES

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Defaults
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_EXERCISE = "wrist_extension"
DEFAULT_TIME_STEPS = 20


class LSTMInference:
    """Real-time LSTM inference for wrist exercise classification"""
    
    def __init__(self, model_path, time_steps=20, device='cpu'):
        self.time_steps = time_steps
        self.device = torch.device(device)
        
        # Load model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = PatientActivityLSTM(
            input_size=TOTAL_FEATURES,
            hidden_size=50,
            num_layers=4,
            num_classes=len(CLASS_NAMES),
            dropout=0.5
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.class_names = checkpoint.get('class_names', CLASS_NAMES)
        
        print(f"✓ Model loaded successfully")
        print(f"  Validation accuracy: {checkpoint['val_acc']*100:.2f}%")
        print(f"  Time steps: {checkpoint.get('time_steps', time_steps)}")
        
        # Feature buffer for time-step sequences
        self.feature_buffer = deque(maxlen=time_steps)
        
    def extract_features_from_landmarks(self, landmarks):
        """
        Extract 132 features from MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
        
        Returns:
            features: numpy array of shape (132,)
        """
        if landmarks is None:
            return None
        
        features = []
        for landmark in landmarks.landmark:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, features):
        """
        Make prediction on current feature buffer.
        
        Args:
            features: Single frame features (132,)
        
        Returns:
            prediction: Class label
            confidence: Confidence score
            class_probs: Probability distribution over classes
        """
        if features is None:
            return None, 0.0, None
        
        # Add to buffer
        self.feature_buffer.append(features)
        
        # Need full buffer for prediction
        if len(self.feature_buffer) < self.time_steps:
            return None, 0.0, None
        
        # Create sequence tensor
        sequence = np.array(list(self.feature_buffer), dtype=np.float32)  # (time_steps, 132)
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)  # (1, time_steps, 132)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        class_probs = probs.cpu().numpy()[0]
        
        return predicted_class, confidence_score, class_probs
    
    def reset_buffer(self):
        """Reset the feature buffer"""
        self.feature_buffer.clear()


def test_on_video(video_path, model_path, time_steps=20, device='cpu', 
                  save_output=False, output_path=None):
    """
    Test LSTM model on a video file.
    
    Args:
        video_path: Path to input video
        model_path: Path to trained model
        time_steps: Time steps K
        device: 'cpu' or 'cuda'
        save_output: Whether to save output video
        output_path: Path to save output video
    """
    # Initialize inference
    lstm_inference = LSTMInference(model_path, time_steps, device)
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    )
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nProcessing video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Video writer
    writer = None
    if save_output and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    predictions_history = []
    processing_times = []
    
    print("\nProcessing frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process with MediaPipe
        results = pose.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract features and predict
        predicted_class = None
        confidence = 0.0
        
        if results.pose_landmarks:
            features = lstm_inference.extract_features_from_landmarks(results.pose_landmarks)
            predicted_class, confidence, class_probs = lstm_inference.predict(features)
            
            # Draw skeleton
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Processing time
        proc_time = time.time() - start_time
        processing_times.append(proc_time)
        
        # Draw prediction
        if predicted_class is not None:
            class_name = lstm_inference.class_names[predicted_class]
            predictions_history.append(predicted_class)
            
            # Color based on class
            if class_name == "good":
                color = (0, 255, 0)  # Green
            elif class_name == "bad":
                color = (0, 0, 255)  # Red
            else:  # mediocre
                color = (0, 165, 255)  # Orange
            
            # Draw prediction box
            cv2.rectangle(image, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (400, 120), color, 2)
            
            cv2.putText(image, f"Prediction: {class_name.upper()}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(image, f"Confidence: {confidence*100:.1f}%", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Buffering message
            cv2.rectangle(image, (10, 10), (400, 80), (0, 0, 0), -1)
            cv2.putText(image, "Buffering frames...", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Draw FPS
        fps_text = f"FPS: {1/proc_time:.1f}"
        cv2.putText(image, fps_text, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw progress
        progress = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(image, progress, (width - 250, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Write frame
        if writer:
            writer.write(image)
        
        # Display
        cv2.imshow('LSTM Exercise Classification', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    pose.close()
    
    # Statistics
    if predictions_history:
        print(f"\n{'='*50}")
        print("RESULTS")
        print(f"{'='*50}")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {1/np.mean(processing_times):.1f}")
        print(f"Average processing time: {np.mean(processing_times)*1000:.1f} ms/frame")
        
        # Prediction distribution
        unique, counts = np.unique(predictions_history, return_counts=True)
        print(f"\nPrediction distribution:")
        for cls, count in zip(unique, counts):
            class_name = lstm_inference.class_names[cls]
            percentage = count / len(predictions_history) * 100
            print(f"  {class_name}: {count} frames ({percentage:.1f}%)")
        
        # Majority vote
        majority_class = unique[np.argmax(counts)]
        majority_name = lstm_inference.class_names[majority_class]
        majority_percent = np.max(counts) / len(predictions_history) * 100
        print(f"\nOverall classification: {majority_name.upper()} ({majority_percent:.1f}%)")
        
        if save_output and output_path:
            print(f"\nOutput saved to: {output_path}")


def test_on_webcam(model_path, time_steps=20, device='cpu', camera_id=0):
    """
    Test LSTM model on live webcam feed.
    
    Args:
        model_path: Path to trained model
        time_steps: Time steps K
        device: 'cpu' or 'cuda'
        camera_id: Camera device ID
    """
    # Initialize inference
    lstm_inference = LSTMInference(model_path, time_steps, device)
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    )
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    print(f"\n{'='*50}")
    print("REAL-TIME WEBCAM CLASSIFICATION")
    print(f"{'='*50}")
    print("Press 'q' to quit")
    print("Press 'r' to reset buffer")
    print(f"{'='*50}\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process with MediaPipe
        results = pose.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract features and predict
        predicted_class = None
        confidence = 0.0
        
        if results.pose_landmarks:
            features = lstm_inference.extract_features_from_landmarks(results.pose_landmarks)
            predicted_class, confidence, class_probs = lstm_inference.predict(features)
            
            # Draw skeleton
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Processing time
        proc_time = time.time() - start_time
        
        # Get frame dimensions
        height, width = image.shape[:2]
        
        # Draw prediction
        if predicted_class is not None:
            class_name = lstm_inference.class_names[predicted_class]
            
            # Color based on class
            if class_name == "good":
                color = (0, 255, 0)  # Green
            elif class_name == "bad":
                color = (0, 0, 255)  # Red
            else:  # mediocre
                color = (0, 165, 255)  # Orange
            
            # Draw prediction box
            cv2.rectangle(image, (10, 10), (450, 150), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (450, 150), color, 3)
            
            cv2.putText(image, f"Exercise: {class_name.upper()}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(image, f"Confidence: {confidence*100:.1f}%", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Draw probability bars
            if class_probs is not None:
                bar_y = 180
                for i, (cls_name, prob) in enumerate(zip(lstm_inference.class_names.values(), class_probs)):
                    bar_width = int(prob * 200)
                    bar_color = (0, 255, 0) if i == predicted_class else (100, 100, 100)
                    
                    cv2.rectangle(image, (20, bar_y), (20 + bar_width, bar_y + 20), bar_color, -1)
                    cv2.putText(image, f"{cls_name}: {prob*100:.1f}%", 
                               (230, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    bar_y += 30
        else:
            # Buffering message
            buffer_size = len(lstm_inference.feature_buffer)
            cv2.rectangle(image, (10, 10), (450, 80), (0, 0, 0), -1)
            cv2.putText(image, f"Buffering: {buffer_size}/{time_steps}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Draw FPS
        fps_text = f"FPS: {1/proc_time:.1f}"
        cv2.putText(image, fps_text, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw instructions
        cv2.putText(image, "Press 'q' to quit | 'r' to reset", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('LSTM Real-time Exercise Classification', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            lstm_inference.reset_buffer()
            print("Buffer reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


def test_on_pickle(pickle_path, model_path, time_steps=20, device='cpu'):
    """
    Test LSTM model on a pickle file.
    
    Args:
        pickle_path: Path to pickle file containing sequence
        model_path: Path to trained model
        time_steps: Time steps K
        device: 'cpu' or 'cuda'
    """
    # Initialize inference
    lstm_inference = LSTMInference(model_path, time_steps, device)
    
    # Load pickle
    print(f"Loading sequence from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        sequence = pickle.load(f)
    
    print(f"Sequence length: {len(sequence)} frames")
    
    # Extract true label from filename
    filename = Path(pickle_path).stem.lower()
    true_label = 1 if "bad" in filename else 2 if "mediocre" in filename else 0
    true_class_name = CLASS_NAMES[true_label]
    
    print(f"True label: {true_class_name}")
    
    # Process sequence
    predictions = []
    confidences = []
    
    for frame in sequence:
        # Extract features
        features = []
        for landmark in frame:
            if len(landmark) == 3:  # (x, y, z)
                features.extend([landmark[0], landmark[1], landmark[2], 1.0])
            elif len(landmark) >= 4:  # (x, y, z, visibility, ...)
                features.extend(landmark[:4])
            else:
                features.extend([0, 0, 0, 0])
        
        features = np.array(features, dtype=np.float32)
        
        # Predict
        pred_class, confidence, _ = lstm_inference.predict(features)
        
        if pred_class is not None:
            predictions.append(pred_class)
            confidences.append(confidence)
    
    # Results
    if predictions:
        print(f"\n{'='*50}")
        print("RESULTS")
        print(f"{'='*50}")
        print(f"Frames processed: {len(predictions)}")
        
        # Prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"\nPrediction distribution:")
        for cls, count in zip(unique, counts):
            class_name = CLASS_NAMES[cls]
            percentage = count / len(predictions) * 100
            print(f"  {class_name}: {count} frames ({percentage:.1f}%)")
        
        # Majority vote
        majority_class = unique[np.argmax(counts)]
        majority_name = CLASS_NAMES[majority_class]
        majority_percent = np.max(counts) / len(predictions) * 100
        avg_confidence = np.mean(confidences)
        
        print(f"\nFinal prediction: {majority_name.upper()} ({majority_percent:.1f}%)")
        print(f"Average confidence: {avg_confidence*100:.1f}%")
        print(f"True label: {true_class_name.upper()}")
        
        # Check correctness
        if majority_class == true_label:
            print("\n✓ CORRECT PREDICTION!")
        else:
            print("\n✗ INCORRECT PREDICTION")


def main():
    parser = argparse.ArgumentParser(description="Test LSTM wrist exercise classifier")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Directory containing trained models")
    parser.add_argument("--exercise", type=str, default=DEFAULT_EXERCISE,
                        choices=["wrist_extension", "wrist_flexion"],
                        help="Exercise type")
    parser.add_argument("--time_steps", type=int, default=DEFAULT_TIME_STEPS,
                        help="Time steps K (should match training)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use")
    
    # Test mode
    parser.add_argument("--mode", type=str, default="webcam",
                        choices=["webcam", "video", "pickle"],
                        help="Test mode")
    parser.add_argument("--input", type=str, default=None,
                        help="Input video/pickle file (for video/pickle mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video file (for video mode)")
    parser.add_argument("--camera_id", type=int, default=0,
                        help="Camera ID (for webcam mode)")
    
    args = parser.parse_args()
    
    # Model path
    model_path = args.model_dir / args.exercise / f"lstm_{args.exercise}_model.pth"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_lstm.py")
        return
    
    # Run test based on mode
    if args.mode == "webcam":
        test_on_webcam(model_path, args.time_steps, args.device, args.camera_id)
    
    elif args.mode == "video":
        if not args.input:
            print("Error: --input required for video mode")
            return
        
        save_output = args.output is not None
        test_on_video(args.input, model_path, args.time_steps, args.device,
                     save_output, args.output)
    
    elif args.mode == "pickle":
        if not args.input:
            print("Error: --input required for pickle mode")
            return
        
        test_on_pickle(args.input, model_path, args.time_steps, args.device)


if __name__ == "__main__":
    main()
