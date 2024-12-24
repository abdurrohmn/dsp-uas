import cv2
import numpy as np
import time
from collections import deque
from services.camera.webcam import Webcam
from services.detection.face_detection import FaceDetector
from services.processing.rppg import RPPG
from services.visualization.plot_signals_rppg import SignalPlotter
import matplotlib.pyplot as plt

def main():
    """
    Fungsi utama yang menjalankan alur proses rPPG secara real-time.
    
    Proses meliputi pengambilan video dari webcam, deteksi wajah, ekstraksi forehead ROI,
    pengolahan sinyal RGB untuk mendapatkan sinyal rPPG, perhitungan detak jantung,
    dan visualisasi hasil secara real-time.
    """
    # Inisialisasi komponen
    webcam = Webcam(fps=30)  # Set FPS ke 30
    face_detector = FaceDetector()
    rppg_processor = RPPG(fps=30, max_window_size=300)  # 10 detik pada 30fps
    signal_plotter = SignalPlotter()
    # Performance monitoring
    frame_times = deque(maxlen=30)  # Track 30 frame terakhir
    process_times = deque(maxlen=5)  # Track 5 waktu pemrosesan terakhir
    
    # Inisialisasi variabel
    current_hr = 0
    current_peaks = None
    current_signal = None
    frame_count = 0  # Inisialisasi frame counter
    
    # Tambahkan variabel untuk monitoring FPS
    start_time = time.time()
    last_time = start_time
    fps_report_interval = 10  # Report setiap 10 detik
    last_frame_count = 0
    interval_count = 1  # Track interval number

    # Arrays to store the raw RGB signals over time
    r_values = []
    g_values = []
    b_values = []

    # Variables for final plotting
    latest_rppg_signal = None
    latest_filtered_signal = None
    latest_peaks = None
    latest_hr = None
    
    while True:
        frame_start = time.time()
        
        ret, frame = webcam.read()
        if not ret:  # Ketika ret = False
            break    # Keluar dari loop
            
        frame_count += 1  # Increment counter untuk setiap frame baru
        
        # Mendeteksi wajah dan get ROI
        detections = face_detector.detect(frame)
        if detections:
            detection = detections[0]
            bbox = face_detector.get_bounding_box(detection, frame.shape)
            adjusted_bbox = face_detector.adjust_bbox(bbox)
            roi = face_detector.get_forehead_roi(frame, adjusted_bbox)
            
            if roi.size > 0:
                # Menghitung nilai rata-rata RGB dari ROI
                mean_rgb = np.mean(roi, axis=(0, 1))
                rppg_processor.add_to_buffer(mean_rgb)

                r_values.append(mean_rgb[0])
                g_values.append(mean_rgb[1])
                b_values.append(mean_rgb[2])
                
                # Menggambar kotak ROI
                x, y, w, h = adjusted_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Memproses sinyal yang terakumulasi jika sudah waktunya
                if rppg_processor.should_process():  # Check setiap 2.5 detik
                    process_start = time.time()
                    hr, peaks, signal, _ = rppg_processor.process_buffer()
                    process_time = time.time() - process_start
                    process_times.append(process_time)
                    if hr is not None:
                        current_hr = hr
                        current_peaks = peaks
                        current_signal = signal
                        avg_process_time = np.mean(process_times)
                        print(f"Processed {frame_count} frames, Current Heart Rate: {current_hr:.1f} BPM, Processing Time: {avg_process_time:.3f}s")
                        
                        latest_rppg_signal = signal
                        latest_filtered_signal = _
                        latest_peaks = peaks
                        latest_hr = hr

        # Menghitung FPS
        frame_times.append(time.time() - frame_start)
        fps = 1.0 / np.mean(frame_times)
        
        # Menambahkan FPS ke tampilan
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Monitor FPS setiap interval
        current_time = time.time()
        if current_time - last_time >= fps_report_interval:
            elapsed_time = current_time - last_time
            frames_in_interval = frame_count - last_frame_count
            actual_fps = frames_in_interval / elapsed_time
            start_interval = ((interval_count - 1) * 10) + 1
            end_interval = interval_count * 10
            print(f"Time window: {start_interval}s - {end_interval}s")
            print(f"Actual FPS: {actual_fps:.1f} ({frames_in_interval} frames in {elapsed_time:.1f}s)")
            last_time = current_time
            last_frame_count = frame_count
            interval_count += 1
        
        if current_hr > 0:
            cv2.putText(frame, f"HR: {current_hr:.1f} BPM",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

        # Display frame
        cv2.imshow('Real-time rPPG', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam.release()
    cv2.destroyAllWindows()

    # Show plots after exiting
    if r_values and latest_rppg_signal is not None and latest_filtered_signal is not None:
        fig_rgb = signal_plotter.plot_rgb_signals(r_values, g_values, b_values)
        fig_hr = signal_plotter.plot_heart_rate(latest_rppg_signal, latest_peaks, latest_hr)
        fig_rgb.show()
        fig_hr.show()
        plt.show()

if __name__ == "__main__":
    main()
