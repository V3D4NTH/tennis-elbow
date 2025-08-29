import numpy as np
import math

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

def calculate_exercise_specific_features(landmarks, exercise_type):
    """Calculate features tailored to specific exercises"""
    config = EXERCISE_CONFIG.get(exercise_type, EXERCISE_CONFIG['wrist_extension'])
    
    features = {}
    
    # Primary metric calculation
    if config['primary_metric'] == 'elbow_flexion':
        features['elbow_flexion'] = calculate_elbow_flexion(landmarks, config['key_joints']['elbow'], config['key_joints']['wrist'])
    
    elif config['primary_metric'] == 'wrist_flexion_angle':
        features['wrist_flexion_angle'] = calculate_wrist_flexion_angle(landmarks, config['key_joints']['elbow'], config['key_joints']['wrist'])
    
    # Secondary metrics
    for metric in config['secondary_metrics']:
        if metric == 'wrist_deviation':
            features['wrist_deviation'] = calculate_wrist_deviation(landmarks, config['key_joints']['elbow'], config['key_joints']['wrist'])
        elif metric == 'movement_smoothness':
            features['movement_smoothness'] = calculate_movement_smoothness(landmarks)
        elif metric == 'elbow_stability':
            features['elbow_stability'] = calculate_elbow_stability(landmarks, config['key_joints']['elbow'])
        elif metric == 'range_of_motion':
            features['range_of_motion'] = calculate_range_of_motion(landmarks, config['key_joints']['elbow'], config['key_joints']['wrist'])
    
    return features

def calculate_wrist_flexion_angle(landmarks, elbow_idx, wrist_idx):
    """Calculate wrist flexion angle for wrist flexion exercises"""
    elbow = landmarks[elbow_idx]
    wrist = landmarks[wrist_idx]
    
    # Vector from elbow to wrist
    v1 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1], wrist[2] - elbow[2]])
    
    # Reference vector (neutral wrist position)
    v2 = np.array([0, -1, 0])  # Pointing downward
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 > 0 and norm_v2 > 0:
        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return math.acos(cos_theta) * 180 / math.pi
    
    return 0