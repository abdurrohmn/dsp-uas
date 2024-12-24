import os  # Mengimpor modul os untuk berinteraksi dengan sistem operasi, seperti manipulasi file dan direktori
import requests  # Mengimpor library requests untuk melakukan permintaan HTTP, digunakan untuk mengunduh model
from tqdm import tqdm  # Mengimpor tqdm untuk menampilkan progress bar saat mengunduh file
import platform  # Mengimpor modul platform untuk mendapatkan informasi tentang sistem operasi
import subprocess  # Mengimpor modul subprocess untuk menjalankan perintah shell dari dalam Python

def download_model():
    """
    Mengunduh model pose landmarker dari MediaPipe.
    Model ini digunakan untuk mendeteksi pose tubuh secara real-time.
    
    Returns:
        str: Path lengkap ke file model yang diunduh.
    """
    model_dir = "models"  # Menetapkan nama direktori tempat model akan disimpan
    os.makedirs(model_dir, exist_ok=True)  # Membuat direktori 'models' jika belum ada. Jika sudah ada, tidak melakukan apa-apa.
    
    # URL tempat model pose landmarker di-host. URL ini mengarah ke model terbaru dengan format float16.
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    
    # Menetapkan nama file lokal tempat model akan disimpan
    filename = os.path.join(model_dir, "pose_landmarker.task")
    
    # Memeriksa apakah file model sudah ada dan tidak kosong
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Model file {filename} already exists and is valid, skipping download.")  # Informasi bahwa file sudah ada
        return filename  # Mengembalikan path file tanpa mengunduh ulang
    
    try:
        print(f"Downloading model to {filename}...")  # Informasi bahwa proses pengunduhan dimulai
        response = requests.get(url, stream=True)  # Mengirim permintaan GET ke URL model dengan streaming aktif
        response.raise_for_status()  # Memeriksa apakah permintaan berhasil (status kode 200). Jika tidak, akan memicu HTTPError
        
        # Mendapatkan total ukuran file dari header 'Content-Length'. Jika tidak tersedia, default ke 0.
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # Menetapkan ukuran blok untuk membaca data dalam 1 KB
    
        # Membuka file untuk ditulis dalam mode binary ('wb') dan menyiapkan progress bar
        with open(filename, 'wb') as f, tqdm(
            total=total_size,  # Total ukuran file untuk progress bar
            unit='iB',  # Satuan unit (byte)
            unit_scale=True,  # Mengaktifkan skala otomatis unit (KB, MB, dll.)
            unit_divisor=1024,  # Faktor pembagi unit (1024 untuk byte)
        ) as pbar:
            # Iterasi melalui setiap blok data yang diterima dari permintaan streaming
            for data in response.iter_content(block_size):
                size = f.write(data)  # Menulis data ke file dan mendapatkan jumlah byte yang ditulis
                pbar.update(size)  # Memperbarui progress bar dengan jumlah byte yang ditulis
    
        # Setelah pengunduhan, memeriksa apakah file yang diunduh tidak kosong
        if os.path.getsize(filename) == 0:
            raise ValueError("Downloaded file is empty")  # Memicu error jika file kosong
        
        print("Download completed successfully!")  # Informasi bahwa pengunduhan selesai
        return filename  # Mengembalikan path file yang telah diunduh
        
    except Exception as e:
        print(f"Error downloading the model: {e}")  # Mencetak pesan error jika terjadi masalah selama pengunduhan
        if os.path.exists(filename):
            os.remove(filename)  # Menghapus file yang mungkin rusak atau tidak lengkap
        raise  # Mengulangi exception untuk ditangani di tingkat yang lebih tinggi

def check_gpu():
    """
    Memeriksa ketersediaan GPU pada sistem.
    
    Returns:
        str: "NVIDIA" untuk GPU NVIDIA, "MLX" untuk Apple Silicon, atau "CPU" jika tidak ada GPU yang tersedia.
    """
    system = platform.system()  # Mendapatkan nama sistem operasi, seperti 'Windows', 'Linux', atau 'Darwin' (macOS)
    print(f"System: {system}")  # Menampilkan sistem operasi yang terdeteksi
    
    # Memeriksa apakah sistem operasi adalah Windows
    if system == "Windows":
        # MediaPipe GPU delegate belum support di Windows
        print("MediaPipe GPU Delegate is not yet supported on Windows. Fallback to CPU.")  # Informasi bahwa GPU tidak didukung di Windows
        return "CPU"  # Mengembalikan "CPU" karena GPU tidak didukung
    
    # Memeriksa apakah sistem operasi adalah Linux
    if system == "Linux":
        try:
            # Menjalankan perintah 'nvidia-smi' untuk memeriksa apakah GPU NVIDIA tersedia
            _ = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            return "NVIDIA"  # Jika perintah berhasil, mengembalikan "NVIDIA"
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Jika perintah gagal atau tidak ditemukan, mengembalikan "CPU"
            return "CPU"
    
    # Memeriksa apakah sistem operasi adalah Darwin (macOS)
    elif system == "Darwin":  # macOS
        try:
            # Mendapatkan informasi brand CPU menggunakan perintah 'sysctl'
            cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
            print(f"CPU: {cpu_info}")  # Menampilkan informasi CPU
            if "Apple" in cpu_info:  # Memeriksa apakah CPU adalah Apple Silicon
                return "MLX"  # Mengembalikan "MLX" untuk Apple Silicon (GPU Apple)
        except subprocess.CalledProcessError:
            pass  # Jika perintah gagal, tidak melakukan apa-apa
    
    return "CPU"  # Jika tidak ada kondisi di atas yang terpenuhi, mengembalikan "CPU"