import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from services.camera.webcam import Webcam
from services.detection.face_detection import FaceDetector
from services.processing.rppg import RPPG
from services.visualization.plot_signals import SignalPlotter

def main():
    # inisiaisasi komponen
    camera = Webcam(source=0, fps=30, resolution=(640, 480))
    face_detector = FaceDetector(model_selection=1, min_detection_confidence=0.5)
    rppg_processor = RPPG(fps=30)
    signal_plotter = SignalPlotter()

    # Penyimpanan sinyal yang ditingkatkan dengan moving average
    window_size = 90  # 3 detik pada 30 fps
    r_signal, g_signal, b_signal = [], [], []
    signal_buffer = deque(maxlen=window_size)
    f_count = 0
    
    # Metrik kualitas sinyal
    min_frames = 150  # Jumlah frame minimum untuk pemrosesan
    signal_quality_threshold = 0.1

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Tidak dapat membaca frame dari webcam.")
                break

            # Terapkan Gaussian blur untuk mengurangi noise
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            # Deteksi wajah
            detections = face_detector.detect(frame)

            if detections:
                detection = detections[0]  # Gunakan wajah pertama yang terdeteksi
                bbox = face_detector.get_bounding_box(detection, frame.shape)
                new_x, new_y, new_width, new_height = face_detector.adjust_bbox(bbox)

                # Pastikan bbox berada dalam batas frame
                new_x = max(0, min(new_x, frame.shape[1] - new_width))
                new_y = max(0, min(new_y, frame.shape[0] - new_height))

                # Gambar bounding box
                cv2.rectangle(frame, (new_x, new_y), 
                            (new_x + new_width, new_y + new_height), 
                            (0, 255, 0), 2)

                # Dapatkan ROI dan hitung rata-rata RGB
                roi = frame[new_y:new_y + new_height, new_x:new_x + new_width]
                
                # Gunakan ROI forehead untuk kualitas sinyal yang lebih baik
                forehead_roi = face_detector.get_forehead_roi(frame, (new_x, new_y, new_width, new_height))
                
                if forehead_roi.size > 0:
                    # Hitung nilai rata-rata RGB
                    rgb_means = cv2.mean(forehead_roi)[:3]
                    signal_buffer.append(rgb_means)

                    # Terapkan moving average untuk menghaluskan sinyal
                    if len(signal_buffer) == window_size:
                        averaged_signals = np.mean(signal_buffer, axis=0)
                        r_signal.append(averaged_signals[2])  # BGR ke RGB
                        g_signal.append(averaged_signals[1])
                        b_signal.append(averaged_signals[0])

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

    # Proses sinyal hanya jika datanya cukup
    if len(r_signal) >= min_frames:
        # Detrend dan normalisasi sinyal
        r_signal = np.array(r_signal)
        g_signal = np.array(g_signal)
        b_signal = np.array(b_signal)

        # Proses rPPG
        rppg_signal = rppg_processor.process_pos([r_signal, g_signal, b_signal])
        filtered_signal = rppg_processor.filter_signal(rppg_signal)
        heart_rate, peaks, normalized_signal = rppg_processor.compute_heart_rate(filtered_signal)

        # Plot hasil
        signal_plotter.plot_rgb_signals(r_signal, g_signal, b_signal)
        signal_plotter.plot_rppg_comparison(rppg_signal, filtered_signal)
        signal_plotter.plot_heart_rate(normalized_signal, peaks, heart_rate)
        plt.show()

if __name__ == '__main__':
    main()
