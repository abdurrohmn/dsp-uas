# main.py

import cv2
import numpy as np
from services.camera.webcam import Webcam
from services.detection.face_detection import FaceDetector

def main():
    # Inisialisasi kamera
    camera = Webcam(source=0, fps=30, resolution=(640, 480))  # Menggunakan webcam default
    face_detector = FaceDetector(model_selection=1, min_detection_confidence=0.5)

    # Variabel untuk sinyal RGB
    r_signal, g_signal, b_signal = [], [], []
    f_count = 0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Tidak dapat membaca frame dari webcam.")
                break

            # Deteksi wajah
            detections = face_detector.detect(frame)

            for detection in detections:
                # Dapatkan bounding box
                bbox = face_detector.get_bounding_box(detection, frame.shape)
                new_x, new_y, new_width, new_height = face_detector.adjust_bbox(bbox, size_from_center=70)

                # Pastikan koordinat bounding box berada dalam batas frame
                new_x = max(new_x, 0)
                new_y = max(new_y, 0)
                new_width = min(new_width, frame.shape[1] - new_x)
                new_height = min(new_height, frame.shape[0] - new_y)

                # Gambar bounding box
                cv2.rectangle(frame, (new_x, new_y), (new_x + new_width, new_y + new_height), (0, 255, 0), 2)

                # Ekstrak ROI dan hitung rata-rata RGB
                roi = frame[new_y:new_y + new_height, new_x:new_x + new_width]
                r_signal.append(np.mean(roi[:, :, 0]))
                g_signal.append(np.mean(roi[:, :, 1]))
                b_signal.append(np.mean(roi[:, :, 2]))

            # Tampilkan frame
            cv2.imshow('Webcam Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Pengguna menghentikan proses.")
                break

            f_count += 1

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    # Pastikan ada sinyal yang dikumpulkan
    if not r_signal or not g_signal or not b_signal:
        print("Tidak ada sinyal RGB yang dikumpulkan.")
        return

    # ... (lanjutan proses setelah loop)

if __name__ == '__main__':
    main()
