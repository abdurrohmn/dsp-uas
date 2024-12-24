import cv2  # Mengimpor library OpenCV untuk pemrosesan gambar dan video
import matplotlib  # Mengimpor library Matplotlib untuk pembuatan plot
matplotlib.use('Agg')  # Gunakan backend Agg agar dapat render plot ke buffer tanpa memerlukan tampilan GUI
import matplotlib.pyplot as plt  # Mengimpor modul pyplot dari Matplotlib untuk pembuatan plot
import numpy as np  # Mengimpor library NumPy untuk manipulasi array dan operasi numerik

def render_plot_to_image(time_data, disp_data, width=400, height=300, peaks=None, troughs=None):
    """
    Membuat atau memperbarui plot (time vs. displacement) menggunakan matplotlib,
    lalu me-render hasil plot ke buffer (array NumPy) agar bisa di-overlay.
    Jika peaks dan troughs diberikan, mereka akan ditandai pada plot.

    Parameters:
    - time_data (list atau array): Data waktu yang akan diplot pada sumbu X.
    - disp_data (list atau array): Data displacement yang akan diplot pada sumbu Y.
    - width (int): Lebar plot dalam piksel (default: 400).
    - height (int): Tinggi plot dalam piksel (default: 300).
    - peaks (list atau array, optional): Indeks data yang merupakan puncak.
    - troughs (list atau array, optional): Indeks data yang merupakan lembah.

    Returns:
    - data (array NumPy): Gambar plot yang di-render dalam format RGB.
    """
    # Membuat figure Matplotlib dengan ukuran yang ditentukan dan resolusi 100 DPI
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111)  # Menambahkan subplot tunggal ke figure
    
    # Menggambar data displacement terhadap waktu dengan garis merah
    ax.plot(time_data, disp_data, color='red', linewidth=2, label='Avg Y Displacement')
    
    # Jika ada puncak yang diberikan, tandai dengan 'x' berwarna biru
    if peaks is not None and len(peaks) > 0:
        ax.plot(time_data[peaks], disp_data[peaks], "x", color='blue', label='Peaks')
    
    # Jika ada lembah yang diberikan, tandai dengan 'o' berwarna hijau
    if troughs is not None and len(troughs) > 0:
        ax.plot(time_data[troughs], disp_data[troughs], "o", color='green', label='Troughs')
    
    # Menetapkan judul dan label sumbu
    ax.set_title("Shoulder Movement", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Avg Y Displacement")
    
    ax.grid(True)  # Menampilkan grid pada plot
    ax.legend(loc='upper right')  # Menampilkan legenda di pojok kanan atas
    
    # Render figure ke canvas Matplotlib
    fig.canvas.draw()
    
    # Mengambil data RGBA dari canvas dan mengonversinya menjadi array NumPy
    data = np.array(fig.canvas.buffer_rgba())  # Bentuk: (height, width, 4)
    
    plt.close(fig)  # Menutup figure untuk membebaskan memori
    
    # Mengonversi data dari format RGBA ke RGB menggunakan OpenCV
    data = cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)
    return data  # Mengembalikan gambar plot dalam format RGB

def overlay_image(frame, overlay_img, x_offset=0, y_offset=0):
    """
    Menempelkan overlay_img (RGB) di atas frame (BGR) pada koordinat (x_offset, y_offset).

    Parameters:
    - frame (array NumPy): Gambar utama dalam format BGR.
    - overlay_img (array NumPy): Gambar overlay dalam format RGB.
    - x_offset (int): Posisi horizontal (X) untuk menempatkan overlay (default: 0).
    - y_offset (int): Posisi vertikal (Y) untuk menempatkan overlay (default: 0).

    Returns:
    - frame (array NumPy): Gambar utama dengan overlay yang telah ditempel.
    """
    # Konversi overlay dari RGB ke BGR agar sesuai dengan format frame
    overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
    
    h, w, _ = overlay_bgr.shape  # Mendapatkan tinggi dan lebar overlay
    rows, cols, _ = frame.shape  # Mendapatkan tinggi dan lebar frame
    
    # Menyesuaikan tinggi overlay jika melebihi batas frame
    if y_offset + h > rows:
        h = rows - y_offset  # Mengurangi tinggi overlay agar tidak melebihi batas
        overlay_bgr = overlay_bgr[:h, :, :]  # Memotong overlay sesuai tinggi baru
    
    # Menyesuaikan lebar overlay jika melebihi batas frame
    if x_offset + w > cols:
        w = cols - x_offset  # Mengurangi lebar overlay agar tidak melebihi batas
        overlay_bgr = overlay_bgr[:, :w, :]  # Memotong overlay sesuai lebar baru
    
    # Mendefinisikan region of interest (ROI) pada frame di mana overlay akan ditempel
    roi = frame[y_offset:y_offset+h, x_offset:x_offset+w]
    
    # Menempelkan overlay ke ROI pada frame
    frame[y_offset:y_offset+h, x_offset:x_offset+w] = overlay_bgr
    return frame  # Mengembalikan frame yang telah di-overlay

def plot_analysis(time_data, disp_data):
    """
    Membuat dan menyimpan plot analisis setelah proses real-time selesai.
    Plot yang dibuat:
      - Respiratory Signal (Time vs Avg Y Displacement)
      - Respiratory Signal Distribution (Histogram)

    Parameters:
    - time_data (list atau array): Data waktu untuk plot sinyal pernapasan.
    - disp_data (list atau array): Data displacement untuk plot sinyal pernapasan.
    """
    # Plot 1: Respiratory Signal (Time vs Avg Y Displacement)
    plt.figure(figsize=(10, 4))  # Membuat figure dengan ukuran 10x4 inci
    plt.plot(time_data, disp_data, color='red', linewidth=2, label='Avg Y Displacement')  # Menggambar sinyal pernapasan
    plt.title("Respiratory Signal")  # Menetapkan judul plot
    plt.xlabel("Time (s)")  # Label sumbu X
    plt.ylabel("Avg Y Displacement")  # Label sumbu Y
    plt.grid(True)  # Menampilkan grid
    plt.legend()  # Menampilkan legenda
    plt.tight_layout()  # Menyesuaikan layout agar tidak terpotong
    plt.savefig("respiratory_signal.png")  # Menyimpan plot sebagai file PNG
    print("Saved plot: respiratory_signal.png")  # Menampilkan pesan bahwa plot telah disimpan
    plt.close()  # Menutup figure untuk membebaskan memori

    # Plot 2: Respiratory Signal Distribution (Histogram)
    plt.figure(figsize=(6, 4))  # Membuat figure dengan ukuran 6x4 inci
    plt.hist(disp_data, bins=30, color='blue', edgecolor='black', alpha=0.7)  # Membuat histogram dari data displacement
    plt.title("Respiratory Signal Distribution")  # Menetapkan judul plot
    plt.xlabel("Avg Y Displacement")  # Label sumbu X
    plt.ylabel("Frequency")  # Label sumbu Y
    plt.grid(True)  # Menampilkan grid
    plt.tight_layout()  # Menyesuaikan layout agar tidak terpotong
    plt.savefig("respiratory_signal_distribution.png")  # Menyimpan plot sebagai file PNG
    print("Saved plot: respiratory_signal_distribution.png")  # Menampilkan pesan bahwa plot telah disimpan
    plt.close()  # Menutup figure untuk membebaskan memori

    # Menampilkan hasil plot dengan OpenCV (opsional)
    try:
        # Membaca gambar plot yang telah disimpan menggunakan OpenCV
        signal_img = cv2.imread("respiratory_signal.png")  # Membaca gambar sinyal pernapasan
        distribution_img = cv2.imread("respiratory_signal_distribution.png")  # Membaca gambar distribusi sinyal pernapasan

        # Memeriksa apakah kedua gambar berhasil dibaca
        if signal_img is not None and distribution_img is not None:
            # Menampilkan gambar sinyal pernapasan dalam jendela bernama "Respiratory Signal"
            cv2.imshow("Respiratory Signal", signal_img)
            # Menampilkan gambar distribusi sinyal pernapasan dalam jendela bernama "Respiratory Signal Distribution"
            cv2.imshow("Respiratory Signal Distribution", distribution_img)
            print("Press any key on the plot windows to exit.")  # Instruksi untuk keluar dari jendela
            cv2.waitKey(0)  # Menunggu input tombol dari pengguna
            cv2.destroyAllWindows()  # Menutup semua jendela OpenCV yang terbuka
        else:
            print("Could not load the saved plot images for display.")  # Pesan jika gambar gagal dibaca
    except Exception as e:
        print(f"Error displaying plots: {e}")  # Menangani dan mencetak error jika terjadi masalah saat menampilkan plot