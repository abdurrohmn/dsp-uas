import cv2
from service.camera.webcam import Webcam

def main():
    # Inisialisasi kamera
    camera = Webcam(source=0, fps=30, resolution=(640, 480))  # Menggunakan webcam default

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Tidak dapat membaca frame dari webcam.")
                break

            # Tampilkan frame
            cv2.imshow('Webcam Feed', frame)

            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Pengguna menghentikan proses.")
                break

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

    finally:
        camera.release()
        cv2.destroyAllWindows()

if _name_ == '_main_':
    main()