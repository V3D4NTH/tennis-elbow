import numpy as np
import math

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

def calculate_exercise_specific_features(landmarks_sequence, exercise_type):
    """Calculate biomechanical features tailored to specific exercises"""
    config = EXERCISE_CONFIG[exercise_type]
    
    features = {}
    
    # Primary metric calculation
    if config['primary_metric'] == 'elbow_flexion':
        features['elbow_flexion'] = calculate_elbow_flexion(landmarks_sequence, config['key_joints']['elbow'], config['key_joints']['wrist'])
    
    elif config['primary_metric'] == 'wrist_flexion_angle':
        features['wrist_flexion_angle'] = calculate_wrist_flexion_angle(landmarks_sequence, config['key_joints']['elbow'], config['key_joints']['wrist'])
    
    # Secondary metrics
    for metric in config['secondary_metrics']:
        if metric == 'wrist_deviation':
            features['wrist_deviation'] = calculate_wrist_deviation(landmarks_sequence, config['key_joints']['elbow'], config['key_joints']['wrist'])
        elif metric == 'movement_smoothness':
            features['movement_smoothness'] = calculate_movement_smoothness(landmarks_sequence)
        elif metric == 'elbow_stability':
            features['elbow_stability'] = calculate_elbow_stability(landmarks_sequence, config['key_joints']['elbow'])
        elif metric == 'range_of_motion':
            features['range_of_motion'] = calculate_range_of_motion(landmarks_sequence, config['key_joints']['elbow'], config['key_joints']['wrist'])
    
    return features

def calculate_elbow_flexion(landmarks_sequence, elbow_idx, wrist_idx):
    """Calculate elbow flexion angle for wrist extension exercise"""
    shoulder = landmarks_sequence[11]  # Standard shoulder index
    elbow = landmarks_sequence[elbow_idx]
    wrist = landmarks_sequence[wrist_idx]
    
    v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1], shoulder[2] - elbow[2]])
    v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1], wrist[2] - elbow[2]])
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 > 0 and norm_v2 > 0:
        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return math.acos(cos_theta) * 180 / math.pi
    
    return 0

def calculate_wrist_flexion_angle(landmarks_sequence, elbow_idx, wrist_idx):
    """Calculate wrist flexion angle for wrist flexion exercise"""
    elbow = landmarks_sequence[elbow_idx]
    wrist = landmarks_sequence[wrist_idx]
    
    # Vector from elbow to wrist
    v1 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1], wrist[2] - elbow[2]])
    
    # Reference vector (vertical)
    v2 = np.array([0, -1, 0])  # Pointing downward
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    
    if norm_v1 > 0:
        cos_theta = dot_product / norm_v1
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return math.acos(cos_theta) * 180 / math.pi
    
    return 0

def calculate_wrist_deviation(landmarks_sequence, elbow_idx, wrist_idx):
    """Calculate wrist deviation angle"""
    elbow = landmarks_sequence[elbow_idx]
    wrist = landmarks_sequence[wrist_idx]
    
    # Vector from elbow to wrist
    v1 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1], wrist[2] - elbow[2]])
    
    # Reference vector (vertical)
    v2 = np.array([0, 1, 0])  # Pointing upward
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    
    if norm_v1 > 0:
        cos_theta = dot_product / norm_v1
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return math.acos(cos_theta) * 180 / math.pi
    
    return 0

def calculate_movement_smoothness(landmarks_sequence):
    """Calculate movement smoothness based on landmark trajectory"""
    positions = [landmarks[15][:2] for landmarks in landmarks_sequence]
    velocities = np.diff(positions, axis=0)
    accelerations = np.diff(velocities, axis=0)
    
    jerk = np.mean(np.abs(accelerations))
    return jerk

def calculate_elbow_stability(landmarks_sequence, elbow_idx):
    """Calculate elbow stability by analyzing variance in joint position"""
    elbow_positions = [landmarks[elbow_idx][:3] for landmarks in landmarks_sequence]
    elbow_positions = np.array(elbow_positions)
    
    # Calculate standard deviation of elbow position
    std_deviation = np.std(elbow_positions, axis=0)
    overall_stability = np.mean(std_deviation)
    
    return overall_stability

def calculate_range_of_motion(landmarks_sequence, elbow_idx, wrist_idx):
    """Calculate range of motion between elbow and wrist"""
    elbow_positions = [landmarks[elbow_idx][:3] for landmarks in landmarks_sequence]
    wrist_positions = [landmarks[wrist_idx][:3] for landmarks in landmarks_sequence]
    
    # Calculate distances between elbow and wrist
    distances = [np.linalg.norm(np.array(elbow) - np.array(wrist)) 
                 for elbow, wrist in zip(elbow_positions, wrist_positions)]
    
    # Return max distance minus min distance
    return max(distances) - min(distances)