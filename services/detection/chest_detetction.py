# chest_detection.py

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks import BaseOptions
from mediapipe.tasks.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

def initialize_pose_landmarker(model_path, delegate="CPU"):
    """
    Menginisialisasi PoseLandmarker dari MediaPipe dengan opsi yang diberikan.
    
    Args:
        model_path (str): Path ke file model pose landmarker.
        delegate (str): "GPU", "MLX", atau "CPU" untuk akselerasi hardware.
        
    Returns:
        PoseLandmarker: Objek PoseLandmarker yang telah diinisialisasi.
    """
    base_options = BaseOptions(model_asset_path=model_path)
    
    if delegate == "GPU":
        base_options.delegate = BaseOptions.Delegate.GPU
    elif delegate == "MLX":
        base_options.delegate = BaseOptions.Delegate.MLX
    else:
        base_options.delegate = BaseOptions.Delegate.CPU
    
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    
    # Create PoseLandmarker
    pose_landmarker = PoseLandmarker.create_from_options(options)
    
    return pose_landmarker
