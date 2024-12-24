import cv2  # Mengimpor library OpenCV untuk pemrosesan gambar dan video
import time  # Mengimpor modul time untuk mengatur jeda waktu
import mediapipe as mp  # Mengimpor library MediaPipe untuk deteksi pose dan landmark

def warmup_camera(cap, num_frames=10):
    """
    Membuang beberapa frame awal dari webcam agar pencahayaan/eksposur
    menyesuaikan terlebih dahulu sebelum mulai diproses.
    
    Parameters:
    - cap: Objek VideoCapture dari OpenCV yang mewakili webcam
    - num_frames: Jumlah frame awal yang akan dibuang (default: 10)
    """
    for i in range(num_frames):  # Mengulangi sebanyak 'num_frames' kali
        ret, _ = cap.read()  # Membaca frame dari kamera
        if not ret:  # Jika gagal membaca frame
            break  # Keluar dari loop

def get_first_valid_frame(cap, landmarker, max_attempts=50, delay=0.1):
    """
    Terus mencoba membaca frame dari kamera hingga pose terdeteksi.
    
    Parameters:
    - cap: Objek VideoCapture dari OpenCV yang mewakili webcam
    - landmarker: Objek detektor pose dari MediaPipe
    - max_attempts: Jumlah percobaan maksimal (default: 50)
    - delay: Jeda antar percobaan dalam detik (default: 0.1)
    
    Returns:
    - frame: Frame pertama yang berhasil mendeteksi pose
    """
    attempts = 0  # Inisialisasi jumlah percobaan
    while attempts < max_attempts:  # Loop hingga mencapai 'max_attempts'
        ret, frame = cap.read()  # Membaca frame dari kamera
        if not ret:  # Jika gagal membaca frame
            continue  # Lewati iterasi ini dan coba lagi

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mengubah warna frame dari BGR ke RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # Membungkus frame dalam objek MediaPipe Image
        detection_result = landmarker.detect(mp_image)  # Mendeteksi pose pada frame

        # Jika ada pose landmarks yang terdeteksi, kembalikan frame ini
        if detection_result.pose_landmarks:
            print("Pose detected in initial frame.")  # Cetak pesan bahwa pose terdeteksi
            return frame  # Kembalikan frame yang valid

        time.sleep(delay)  # Jeda selama 'delay' detik sebelum mencoba lagi
        attempts += 1  # Increment jumlah percobaan

    # Jika setelah 'max_attempts' percobaan tidak ada pose yang terdeteksi, keluarkan error
    raise ValueError("No pose detected in any of the initial frames!")
