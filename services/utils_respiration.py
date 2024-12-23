# utils_respiration.py

import os
import requests
from tqdm import tqdm
import platform
import subprocess
import cv2
import numpy as np

def download_model():
    """
    Mengunduh model pose landmarker dari MediaPipe.
    Model ini digunakan untuk mendeteksi pose tubuh dalam video.
    """
    # Create models directory if it doesn't exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    filename = os.path.join(model_dir, "pose_landmarker.task")
    
    # Check if file already exists and is not empty
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Model file {filename} already exists and is valid, skipping download.")
        return filename
    
    # Download with progress bar
    try:
        print(f"Downloading model to {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(filename, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        
        # Verify downloaded file
        if os.path.getsize(filename) == 0:
            raise ValueError("Downloaded file is empty")
            
        print("Download completed successfully!")
        return filename
        
    except Exception as e:
        print(f"Error downloading the model: {e}")
        if os.path.exists(filename):
            os.remove(filename)  # Clean up partial download
        raise

def check_gpu():
    """
    Memeriksa ketersediaan GPU pada sistem.
    Returns:
        str: "NVIDIA" untuk GPU NVIDIA, "MLX" untuk Apple Silicon, atau "CPU" jika tidak ada GPU
    """
    system = platform.system()
    print(f"System: {system}")
    # Check for NVIDIA GPU
    if system in ["Linux", "Windows"]:
        try:
            nvidia_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            return "NVIDIA"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "CPU"

    # Check for Apple MLX
    elif system == "Darwin":  # macOS
        try:
            cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            print(f"CPU: {cpu_info}")
            if "Apple" in cpu_info:  # This indicates Apple Silicon (M1/M2/M3)
                return "MLX"
        except subprocess.CalledProcessError:
            pass
    return "CPU"

def enhance_roi(roi):
    """
    Meningkatkan kualitas Region of Interest (ROI) dengan teknik image processing.
    Args:
        roi: Region of Interest dari frame video
    Returns:
        numpy.ndarray: ROI yang telah ditingkatkan kualitasnya
    """
    if roi is None or roi.size == 0:
        raise ValueError("Empty ROI provided")
        
    # Convert to grayscale
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Apply edge enhancement
        enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
        return enhanced
    except cv2.error as e:
        raise ValueError(f"Error processing ROI: {str(e)}")

def get_initial_roi(image, landmarker, x_size=100, y_size=150, shift_x=0, shift_y=0):
    """
    Mendapatkan ROI awal berdasarkan posisi bahu menggunakan pose detection.
    Args:
        image: Frame video input
        landmarker: Model pose detector
        x_size: Jarak piksel dari titik tengah ke tepi kiri/kanan
        y_size: Jarak piksel dari titik tengah ke tepi atas/bawah
        shift_x: Pergeseran horizontal kotak (negatif=kiri, positif=kanan)
        shift_y: Pergeseran vertikal kotak (negatif=atas, positif=bawah)
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )
    
    # Detect landmarks
    detection_result = landmarker.detect(mp_image)
    
    if not detection_result.pose_landmarks:
        raise ValueError("No pose detected in first frame!")
        
    landmarks = detection_result.pose_landmarks[0]
    
    # Get shoulder positions
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    
    # Calculate center point between shoulders
    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)
    
    # Apply shifts to center point
    center_x += shift_x
    center_y += shift_y
    
    # Calculate ROI boundaries from center point and sizes
    left_x = max(0, center_x - x_size)
    right_x = min(width, center_x + x_size)
    top_y = max(0, center_y - y_size)
    bottom_y = min(height, center_y + y_size)
    
    # Validate ROI size
    if (right_x - left_x) <= 0 or (bottom_y - top_y) <= 0:
        raise ValueError("Invalid ROI dimensions")
        
    return (left_x, top_y, right_x, bottom_y)

def prepare_plot():
    """
    Mempersiapkan plot matplotlib untuk visualisasi gerakan bahu.
    Returns:
        tuple: (figure, axis) untuk plotting
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    # Create smaller figure with transparent background
    fig = plt.figure(figsize=(4, 3), facecolor='none')
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.7)  # Semi-transparent background
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Shoulder Movement')
    ax.grid(True, alpha=1)  # Lighter grid
    return fig, ax
