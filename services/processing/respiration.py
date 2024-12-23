import cv2
import numpy as np
import mediapipe as mp

def get_initial_roi(image, model_path, x_size, y_size, shift_x, shift_y):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Create a list to store the detection result
    detection_result = []
    
    def callback(result, output_image, timestamp_ms):
        detection_result.append(result)
    
    # Setup landmarker for initial detection
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )
    
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose_detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = pose_detector.detect(mp_image)
        
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            raise ValueError("No pose detected!")

        landmarks = detection_result.pose_landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        center_x = int((left_shoulder.x + right_shoulder.x) * width / 2) + shift_x
        center_y = int((left_shoulder.y + right_shoulder.y) * height / 2) + shift_y

        left_x = max(0, center_x - x_size)
        right_x = min(width, center_x + x_size)
        top_y = max(0, center_y - y_size)
        bottom_y = min(height, center_y + y_size)

        return left_x, top_y, right_x, bottom_y
    
def enhance_roi(roi):
    """Meningkatkan kualitas ROI untuk tracking yang lebih baik"""
    if roi is None or roi.size == 0:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # Apply edge enhancement
    enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
    return enhanced