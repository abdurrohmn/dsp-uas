import cv2
import mediapipe as mp

def get_initial_roi(image, landmarker, x_size=100, y_size=150, shift_x=0, shift_y=0):
    """
    Mendapatkan ROI awal berdasarkan posisi bahu menggunakan pose detection.
    
    Parameters:
        image (array NumPy): Frame gambar dalam format BGR (default format OpenCV).
        landmarker (PoseLandmarker): Objek PoseLandmarker dari MediaPipe untuk deteksi pose.
        x_size (int, optional): Lebar setengah ROI dari pusat bahu (default: 100 piksel).
        y_size (int, optional): Tinggi setengah ROI dari pusat bahu (default: 150 piksel).
        shift_x (int, optional): Pergeseran horizontal ROI (default: 0 piksel).
        shift_y (int, optional): Pergeseran vertikal ROI (default: 0 piksel).
    
    Returns:
        tuple: Koordinat ROI dalam format (left_x, top_y, right_x, bottom_y).
    
    Raises:
        ValueError: Jika tidak ada pose yang terdeteksi atau ROI tidak valid.
    """
    # Mengonversi gambar dari format BGR ke RGB karena MediaPipe mengharapkan input dalam format RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Mendapatkan dimensi gambar: tinggi dan lebar
    height, width = image.shape[:2]
    
    # Membuat objek Image dari MediaPipe dengan format SRGB dan data gambar RGB
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )
    
    # Menggunakan PoseLandmarker untuk mendeteksi pose dalam gambar
    detection_result = landmarker.detect(mp_image)
    
    # Memeriksa apakah ada pose landmarks yang terdeteksi
    if not detection_result.pose_landmarks:
        raise ValueError("No pose detected in initial frame!")  # Memicu error jika tidak ada pose yang terdeteksi

    # Mendapatkan landmarks pose pertama yang terdeteksi
    landmarks = detection_result.pose_landmarks[0]
    
    # Landmark index 11 = left shoulder, 12 = right shoulder
    left_shoulder = landmarks[11]  # Mendapatkan posisi bahu kiri
    right_shoulder = landmarks[12]  # Mendapatkan posisi bahu kanan
    
    # Menghitung titik tengah di antara kedua bahu
    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)  # Konversi koordinat normalisasi ke piksel
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)  # Konversi koordinat normalisasi ke piksel
    
    # Menerapkan pergeseran (shift_x, shift_y) pada titik tengah
    center_x += shift_x  # Menambahkan pergeseran horizontal
    center_y += shift_y  # Menambahkan pergeseran vertikal
    
    # Menghitung batas ROI berdasarkan pusat bahu dan ukuran yang ditentukan
    left_x = max(0, center_x - x_size)  # Pastikan tidak kurang dari 0 (batas kiri gambar)
    right_x = min(width, center_x + x_size)  # Pastikan tidak melebihi lebar gambar (batas kanan)
    top_y = max(0, center_y - y_size)  # Pastikan tidak kurang dari 0 (batas atas gambar)
    bottom_y = min(height, center_y + y_size)  # Pastikan tidak melebihi tinggi gambar (batas bawah)
    
    # Memeriksa validitas dimensi ROI
    if (right_x - left_x) <= 0 or (bottom_y - top_y) <= 0:
        raise ValueError("Invalid ROI dimensions for shoulders!")  # Memicu error jika dimensi ROI tidak valid
    
    # Mengembalikan koordinat ROI sebagai tuple
    return (left_x, top_y, right_x, bottom_y)
