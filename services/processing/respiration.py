import cv2  # Mengimpor library OpenCV untuk pemrosesan gambar dan video
import time  # Mengimpor modul time untuk mengatur jeda waktu dan mengukur durasi
import numpy as np  # Mengimpor library NumPy untuk manipulasi array dan operasi numerik
from scipy.signal import find_peaks, savgol_filter  # Mengimpor fungsi find_peaks dan savgol_filter dari SciPy untuk analisis sinyal

from services.camera.camera_utils import warmup_camera, get_first_valid_frame  # Mengimpor fungsi dari modul custom 'camera_utils'
from services.detection.chest_detetction import get_initial_roi  # Mengimpor fungsi dari modul custom 'roi_utils'
from services.visualization.plot_utils import render_plot_to_image  # Mengimpor fungsi dari modul custom 'plot_utils'

def overlay_image(background, overlay, x_offset, y_offset, alpha=0.5):
    """
    Overlay satu gambar di atas yang lain dengan transparansi.

    Parameters:
        background (array NumPy): Gambar latar belakang dalam format BGR.
        overlay (array NumPy): Gambar overlay dalam format RGB.
        x_offset (int): Koordinat x untuk posisi overlay.
        y_offset (int): Koordinat y untuk posisi overlay.
        alpha (float): Transparansi overlay (0.0 hingga 1.0).

    Returns:
        array NumPy: Gambar hasil overlay.
    """
    bh, bw = background.shape[:2]  # Mendapatkan tinggi (bh) dan lebar (bw) dari gambar latar belakang
    oh, ow = overlay.shape[:2]  # Mendapatkan tinggi (oh) dan lebar (ow) dari gambar overlay

    # Pastikan overlay tidak keluar dari batas frame
    if x_offset + ow > bw or y_offset + oh > bh:
        raise ValueError("Overlay exceeds background dimensions.")  # Memicu error jika overlay melebihi batas frame

    # Ambil bagian dari background yang akan di-overlay
    roi = background[y_offset:y_offset + oh, x_offset:x_offset + ow]  # Mendefinisikan Region of Interest (ROI)

    # Terapkan transparansi menggunakan metode blending
    blended = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)  # Menggabungkan ROI dan overlay dengan tingkat transparansi alpha

    # Tempel blended image kembali ke background
    background[y_offset:y_offset + oh, x_offset:x_offset + ow] = blended  # Mengupdate ROI pada background dengan hasil blending
    return background  # Mengembalikan gambar background yang telah di-overlay

def process_realtime(landmarker, cam_index=0, x_size=300, y_size=250, shift_x=0, shift_y=0):
    """
    Memproses video real-time:
      - Mendapatkan ROI bahu/dada dengan MediaPipe Pose
      - Tracking Optical Flow (Lucas-Kanade)
      - Menampilkan bounding box, titik-titik tracking, serta
        menampilkan grafik real-time displacement di atas video
      - Menghitung dan menampilkan laju pernapasan (breaths per minute)
    Tekan 'q' untuk keluar dari loop.

    Parameters:
        landmarker (PoseLandmarker): Objek PoseLandmarker dari MediaPipe untuk deteksi pose.
        cam_index (int): Index kamera, 0 biasanya merupakan kamera default.
        x_size (int): Lebar ROI (Region of Interest) dalam piksel.
        y_size (int): Tinggi ROI dalam piksel.
        shift_x (int): Pergeseran horizontal ROI.
        shift_y (int): Pergeseran vertikal ROI.

    Returns:
        tuple: (time_data, disp_data) yang dikumpulkan selama proses.
    """
    cap = cv2.VideoCapture(cam_index)  # Membuka kamera dengan index yang ditentukan
    if not cap.isOpened():
        raise ValueError("Cannot open webcam atau perangkat video capture!")  # Memicu error jika kamera gagal dibuka

    # 1. Warm-up camera
    warmup_camera(cap, num_frames=10)  # Membaca dan membuang beberapa frame awal untuk menyesuaikan kamera

    # 2. Frame pertama yang valid
    first_frame = get_first_valid_frame(cap, landmarker, max_attempts=50, delay=0.1)  # Mendapatkan frame pertama yang valid dengan deteksi pose

    # 3. Dapatkan ROI
    roi_coords = get_initial_roi(
        first_frame, landmarker,
        x_size=x_size, y_size=y_size,
        shift_x=shift_x, shift_y=shift_y
    )  # Mendapatkan koordinat ROI awal berdasarkan frame pertama dan deteksi pose
    left_x, top_y, right_x, bottom_y = roi_coords  # Memisahkan koordinat ROI

    # 4. Inisialisasi optical flow
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)  # Mengonversi frame pertama ke grayscale
    roi = old_gray[top_y:bottom_y, left_x:right_x]  # Memotong ROI dari frame grayscale

    # Mendeteksi fitur-fitur yang baik untuk dilacak dalam ROI menggunakan metode Shi-Tomasi
    features = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=60,  # Maksimal 60 fitur
        qualityLevel=0.15,  # Kualitas minimal fitur
        minDistance=3,  # Jarak minimal antar fitur
        blockSize=7  # Ukuran blok untuk perhitungan
    )
    if features is None:
        raise ValueError("No features found to track in initial ROI!")  # Memicu error jika tidak ada fitur yang ditemukan

    features = np.float32(features)  # Mengonversi fitur ke tipe data float32
    # Offset ke koordinat penuh (karena ROI adalah bagian dari frame penuh)
    features[:, :, 0] += left_x  # Menambahkan offset x
    features[:, :, 1] += top_y  # Menambahkan offset y

    # Mengatur parameter untuk algoritma Optical Flow Lucas-Kanade
    lk_params = dict(
        winSize=(15, 15),  # Ukuran window pencarian
        maxLevel=2,  # Maksimal level piramida
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Kriteria berhenti
    )

    start_time = time.time()  # Mencatat waktu mulai
    time_data = []  # List untuk menyimpan data waktu
    disp_data = []  # List untuk menyimpan data displacement
    peaks_indices = []  # List untuk menyimpan indeks puncak
    troughs_indices = []  # List untuk menyimpan indeks lembah

    # Menyimpan waktu puncak atau lembah terakhir yang berhasil dideteksi
    breath_times = []  # List untuk menyimpan waktu deteksi pernapasan
    last_peak_time = 0  # Variabel untuk melacak waktu puncak terakhir

    while True:
        ret, frame = cap.read()  # Membaca frame dari kamera
        if not ret:
            print("Frame read failed. Exiting...")  # Mencetak pesan jika frame gagal dibaca
            break  # Keluar dari loop jika frame gagal dibaca

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengonversi frame saat ini ke grayscale

        # Inisialisasi breaths dan respiratory_rate
        breaths = 0  # Menginisialisasi jumlah napas
        respiratory_rate = 0.0  # Menginisialisasi laju pernapasan

        # Re-calc optical flow
        if features is not None and len(features) > 10:
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, features, None, **lk_params
            )  # Menghitung Optical Flow antara frame lama dan frame baru

            # Mengambil fitur yang berhasil dilacak
            good_old = features[status == 1]  # Fitur yang berhasil dilacak di frame lama
            good_new = new_features[status == 1]  # Fitur yang berhasil dilacak di frame baru

            # Gambar jalur pergerakan (tracking) di frame
            mask = np.zeros_like(frame)  # Membuat mask kosong untuk menggambar jalur
            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()  # Mendapatkan koordinat fitur baru
                c, d = old.ravel()  # Mendapatkan koordinat fitur lama
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)  # Menggambar garis hijau dari fitur lama ke baru
                frame = cv2.circle(frame, (int(a), int(b)), 4, (0, 255, 0), -1)  # Menggambar lingkaran hijau di lokasi fitur baru
            frame = cv2.add(frame, mask)  # Menambahkan mask ke frame untuk menampilkan jalur

            # Update titik-titik fitur yang dilacak
            if len(good_new) > 0:
                features = good_new.reshape(-1, 1, 2)  # Memperbarui fitur dengan posisi baru

            # Hitung displacement (rata-rata sumbu Y)
            y_diff = good_new[:, 1] - good_old[:, 1]  # Menghitung perbedaan posisi pada sumbu Y
            avg_y_diff = np.mean(y_diff) if len(y_diff) > 0 else 0.0  # Menghitung rata-rata perbedaan Y

            current_time = time.time() - start_time  # Menghitung waktu saat ini sejak mulai
            time_data.append(current_time)  # Menambahkan waktu ke list time_data
            disp_data.append(avg_y_diff)  # Menambahkan displacement ke list disp_data

            # ==== Tambahan: Smoothing data dengan Savitzky–Golay ====
            if len(disp_data) >= 9:
                smooth_data = savgol_filter(disp_data, window_length=9, polyorder=3)  # Menghaluskan data displacement
            else:
                smooth_data = disp_data  # Jika data kurang dari window_length, gunakan data asli

            # Deteksi puncak dan lembah
            if len(time_data) >= 2:
                if int(current_time) > int(time_data[-2]):  # Memeriksa jika ada detik baru
                    window_size = 5  # Menetapkan ukuran window untuk deteksi puncak/lembah
                    indices = [i for i, t in enumerate(time_data) if t >= current_time - window_size]  # Mendapatkan indeks data dalam window
                    if len(indices) > 1:
                        recent_disp = np.array([smooth_data[i] for i in indices])  # Mengambil data displacement dalam window
                        peaks, _ = find_peaks(recent_disp, distance=20, prominence=0.01)  # Mendapatkan indeks puncak
                        troughs, _ = find_peaks(-recent_disp, distance=20, prominence=0.01)  # Mendapatkan indeks lembah

                        for peak in peaks:
                            global_peak_idx = indices[peak]  # Mengonversi indeks lokal ke global
                            if global_peak_idx > last_peak_time:  # Memeriksa apakah puncak lebih baru dari yang terakhir
                                breath_times.append(time_data[global_peak_idx])  # Menambahkan waktu puncak ke breath_times
                                peaks_indices.append(global_peak_idx)  # Menambahkan indeks puncak
                                last_peak_time = global_peak_idx  # Memperbarui waktu puncak terakhir

                        for trough in troughs:
                            global_trough_idx = indices[trough]  # Mengonversi indeks lokal ke global
                            if global_trough_idx > last_peak_time:  # Memeriksa apakah lembah lebih baru dari yang terakhir
                                breath_times.append(time_data[global_trough_idx])  # Menambahkan waktu lembah ke breath_times
                                troughs_indices.append(global_trough_idx)  # Menambahkan indeks lembah
                                last_peak_time = global_trough_idx  # Memperbarui waktu lembah terakhir

                        # Menyaring waktu pernapasan yang lebih dari 60 detik yang lalu
                        breath_times = [bt for bt in breath_times if bt >= current_time - 60]

            breaths = len(breath_times) // 2  # Menghitung jumlah napas (puncak + lembah = 2 per napas)
            respiratory_rate = breaths * 60 / 60  # Menghitung laju pernapasan dalam BPM

        else:
            # Jika tidak ada cukup fitur untuk dilacak, coba deteksi kembali ROI dan fitur
            try:
                roi_coords = get_initial_roi(frame, landmarker, x_size, y_size, shift_x, shift_y)  # Mendapatkan ROI baru
                left_x, top_y, right_x, bottom_y = roi_coords  # Memisahkan koordinat ROI
                roi = frame_gray[top_y:bottom_y, left_x:right_x]  # Memotong ROI dari frame grayscale
                features = cv2.goodFeaturesToTrack(
                    roi,
                    maxCorners=60,
                    qualityLevel=0.15,
                    minDistance=3,
                    blockSize=7
                )  # Mendapatkan fitur baru untuk dilacak
                if features is not None:
                    features = np.float32(features)  # Mengonversi fitur ke tipe data float32
                    features[:, :, 0] += left_x  # Menambahkan offset x
                    features[:, :, 1] += top_y  # Menambahkan offset y
            except ValueError as err:
                print(f"Could not re-detect ROI: {err}")  # Mencetak pesan error jika ROI tidak dapat dideteksi ulang
                features = None  # Mengatur fitur menjadi None untuk mencoba deteksi di iterasi berikutnya

        # Menggambar bounding box pada ROI
        cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 2)  # Menggambar rectangle merah di sekitar ROI

        # Mengonversi data waktu dan displacement ke array NumPy
        time_array = np.array(time_data)
        disp_array = np.array(disp_data)

        # Mengonversi indeks puncak dan lembah ke array NumPy
        peaks_np = np.array(peaks_indices)
        troughs_np = np.array(troughs_indices)

        # Memastikan indeks puncak dan lembah tidak melebihi panjang data
        if len(peaks_np) > 0 and np.max(peaks_np) >= len(time_array):
            peaks_np = peaks_np[peaks_np < len(time_array)]
        if len(troughs_np) > 0 and np.max(troughs_np) >= len(time_array):
            troughs_np = troughs_np[troughs_np < len(time_array)]

        # Membuat plot real-time dari data waktu dan displacement
        plot_img = render_plot_to_image(
            time_array, disp_array,
            width=400, height=300,
            peaks=peaks_np,
            troughs=troughs_np
        )  # Menggunakan fungsi custom untuk membuat plot sebagai gambar

        # Menentukan posisi overlay plot pada frame
        x_offset = frame.shape[1] - plot_img.shape[1]  # Posisi horizontal (kanan)
        y_offset = 0  # Posisi vertikal (atas)
        frame = overlay_image(frame, plot_img, x_offset, y_offset, alpha=0.5)  # Menempelkan plot ke frame dengan transparansi

        # Menentukan teks untuk laju pernapasan
        if breaths > 0:
            respiratory_text = f"Respiratory Rate: {respiratory_rate:.1f} BPM"  # Menampilkan BPM jika telah dihitung
        else:
            respiratory_text = "Respiratory Rate: Calculating..."  # Menampilkan pesan sedang menghitung

        # Menentukan warna teks berdasarkan laju pernapasan
        if respiratory_rate < 12:
            color = (0, 0, 255)  # Merah untuk laju rendah
        elif 12 <= respiratory_rate <= 20:
            color = (0, 255, 0)  # Hijau untuk laju normal
        else:
            color = (0, 255, 255)  # Kuning untuk laju tinggi

        # Menambahkan teks laju pernapasan ke frame
        cv2.putText(
            frame,
            respiratory_text,
            (50, 50),  # Posisi teks di frame
            cv2.FONT_HERSHEY_SIMPLEX,  # Font yang digunakan
            1.0,  # Ukuran font
            color,  # Warna teks
            2,  # Ketebalan garis teks
            cv2.LINE_AA  # Tipe garis (anti-aliasing)
        )

        # Menampilkan frame dengan overlay dan teks
        cv2.imshow("Respiration Tracking with Real-time Overlay", frame)  # Menampilkan jendela video

        # Menunggu input tombol 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting loop and generating analysis plots...")  # Mencetak pesan saat keluar
            break  # Keluar dari loop

        old_gray = frame_gray.copy()  # Memperbarui frame lama dengan frame saat ini untuk iterasi berikutnya

    # Setelah loop selesai, melepaskan kamera dan menutup semua jendela OpenCV
    cap.release()  # Menutup kamera
    cv2.destroyAllWindows()  # Menutup semua jendela OpenCV

    return (time_data, disp_data)  # Mengembalikan data waktu dan displacement yang dikumpulkan
