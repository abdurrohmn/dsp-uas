import mediapipe as mp  # Mengimpor library MediaPipe untuk deteksi pose dan landmark
from mediapipe.tasks import python  # Mengimpor submodul 'python' dari MediaPipe Tasks
from mediapipe.tasks.python import vision  # Mengimpor submodul 'vision' dari MediaPipe Tasks Python

from services.utils_respiration import download_model, check_gpu  # Mengimpor fungsi 'download_model' dan 'check_gpu' dari 'model_utils'
from services.camera.camera_utils import warmup_camera, get_first_valid_frame  # Mengimpor fungsi kamera dari 'camera_utils'
from services.detection.chest_detetction import get_initial_roi  # Mengimpor fungsi 'get_initial_roi' dari 'roi_utils'
from services.visualization.plot_utils import plot_analysis  # Mengimpor fungsi 'plot_analysis' dari 'plot_utils'
from services.processing.respiration import process_realtime  # Mengimpor fungsi 'process_realtime' dari 'respiration'

def main():
    """
    Fungsi utama yang menjalankan aplikasi:
    1) Download model pose landmarker
    2) Inisialisasi PoseLandmarker
    3) Jalankan tracking real-time dengan bounding box, titik tracking, dan overlay grafik
    4) Setelah keluar, tampilkan analisis sinyal pernapasan
    """
    detector_image = None  # Inisialisasi variabel 'detector_image' sebagai None
    
    try:
        # 1) Download model pose landmarker
        model_path = download_model()  # Memanggil fungsi 'download_model' untuk mendapatkan path model

        # 2) Siapkan PoseLandmarker
        PoseLandmarker = mp.tasks.vision.PoseLandmarker  # Mendefinisikan kelas PoseLandmarker dari MediaPipe
        BaseOptions = mp.tasks.BaseOptions  # Mendefinisikan kelas BaseOptions dari MediaPipe
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions  # Mendefinisikan kelas PoseLandmarkerOptions dari MediaPipe
        VisionRunningMode = mp.tasks.vision.RunningMode  # Mendefinisikan enum RunningMode dari MediaPipe untuk mode operasi

        gpu_check = check_gpu()  # Memanggil fungsi 'check_gpu' untuk memeriksa jenis GPU yang tersedia
        
        # Menentukan delegate (CPU atau GPU) berdasarkan jenis GPU yang terdeteksi
        if gpu_check == "NVIDIA":
            delegate = BaseOptions.Delegate.GPU  # Menggunakan GPU NVIDIA
        elif gpu_check == "MLX":
            delegate = BaseOptions.Delegate.GPU  # Menggunakan GPU MLX
        else:
            delegate = BaseOptions.Delegate.CPU  # Jika tidak ada GPU yang cocok, gunakan CPU

        # Mengonfigurasi opsi PoseLandmarker
        options_image = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,  # Path ke model pose landmarker
                delegate=delegate  # Delegate yang dipilih (CPU/GPU)
            ),
            running_mode=VisionRunningMode.IMAGE,  # Mode operasi: deteksi single-frame
            num_poses=1,  # Deteksi maksimal 1 pose per frame
            min_pose_detection_confidence=0.5,  # Confidence minimum untuk deteksi pose
            min_pose_presence_confidence=0.5,  # Confidence minimum untuk keberadaan pose
            min_tracking_confidence=0.5,  # Confidence minimum untuk tracking pose
            output_segmentation_masks=False  # Tidak menghasilkan segmentation masks
        )

        # 3) Buat PoseLandmarker dengan opsi yang telah dikonfigurasi
        detector_image = PoseLandmarker.create_from_options(options_image)  # Membuat instance PoseLandmarker

        # 4) Jalankan proses tracking real-time
        print("\nStarting real-time respiration (shoulder/chest) tracking. Tekan 'q' untuk keluar...\n")  # Pesan awal
        time_data, disp_data = process_realtime(
            landmarker=detector_image,  # Objek PoseLandmarker yang digunakan untuk deteksi
            cam_index=0,       # Index kamera, 0 biasanya merupakan kamera default
            x_size=300,        # Ukuran lebar ROI (Region of Interest)
            y_size=200,        # Ukuran tinggi ROI
            shift_x=0,         # Pergeseran horizontal ROI
            shift_y=100        # Pergeseran vertikal ROI
        )
        print("\nReal-time tracking stopped.")  # Pesan setelah tracking berhenti

        # 5) Plot analisis setelah tracking berhenti
        plot_analysis(time_data, disp_data)  # Memanggil fungsi 'plot_analysis' untuk menampilkan hasil analisis

    except Exception as e:
        print(f"An error occurred: {str(e)}")  # Menangani dan mencetak error jika terjadi

    finally:
        if detector_image:
            detector_image.close()  # Menutup PoseLandmarker jika telah dibuat

# Memastikan bahwa fungsi 'main' dipanggil hanya ketika skrip ini dijalankan langsung
if __name__ == "__main__":
    main()  # Memanggil fungsi 'main' untuk menjalankan aplikasi