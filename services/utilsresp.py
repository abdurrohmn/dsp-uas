# src/downloader.py

import os
import requests
import platform
import subprocess
from tqdm import tqdm

def download_model(model_dir="models"):
    """
    Mengunduh model pose landmarker dari MediaPipe.
    Model ini digunakan untuk mendeteksi pose tubuh dalam video.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    filename = os.path.join(model_dir, "pose_landmarker.task")
    
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Model file {filename} already exists and is valid, skipping download.")
        return filename
    
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
        
        if os.path.getsize(filename) == 0:
            raise ValueError("Downloaded file is empty")
            
        print("Download completed successfully!")
        return filename
        
    except Exception as e:
        print(f"Error downloading the model: {e}")
        if os.path.exists(filename):
            os.remove(filename)  # Clean up partial download
        raise

# src/gpu_checker.py
def check_gpu():
    """
    Memeriksa ketersediaan GPU pada sistem.
    Returns:
        str: "NVIDIA" untuk GPU NVIDIA, "MLX" untuk Apple Silicon, atau "CPU" jika tidak ada GPU
    """
    system = platform.system()
    print(f"System: {system}")
    
    if system in ["Linux", "Windows"]:
        try:
            subprocess.check_output(['nvidia-smi'])
            return "NVIDIA"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "CPU"
    elif system == "Darwin":  # macOS
        try:
            cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            print(f"CPU: {cpu_info}")
            if "Apple" in cpu_info:
                return "MLX"
        except subprocess.CalledProcessError:
            pass
    return "CPU"
