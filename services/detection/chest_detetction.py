import cv2
import time
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
import numpy as np
from services.processing.respiration import get_initial_roi, enhance_roi
from services.visualization.plotting_signal_resp import SignalPlotterResp

def setup_pose_landmarker(model_path):
    # Define the callback function that will receive the detection results
    def callback(result: PoseLandmarkerResult,
                output_image: mp.Image,
                timestamp_ms: int):
        pass  # We'll handle results directly in process_frame instead
    
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=model_path,
            delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)

def process_frame(frame, pose_landmarker, timestamps, y_positions):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    detection_result = pose_landmarker.detect(mp_image)
    
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        height, width = frame.shape[:2]
        
        # Hanya gambar landmark bahu (indeks 11 dan 12)
        shoulder_landmarks = [11, 12]  # left and right shoulders
        for idx in shoulder_landmarks:
            landmark = landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Hitung posisi rata-rata bahu
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
        center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)
        
        # Tambahkan optical flow tracking
        roi = frame[center_y-100:center_y+100, center_x-100:center_x+100]
        if roi.size > 0:  # Pastikan ROI valid
            enhanced_roi = enhance_roi(roi)
            # Gambar ROI yang telah dienhance
            frame[center_y-100:center_y+100, center_x-100:center_x+100] = cv2.cvtColor(enhanced_roi, cv2.COLOR_GRAY2BGR)
        
        # Update data untuk plotting
        timestamps.append(time.time())
        y_positions.append(center_y)
        
        # Gambar ROI
        cv2.rectangle(frame, 
                     (center_x-100, center_y-100),
                     (center_x+100, center_y+100),
                     (0, 255, 0), 2)
    
    return frame

def calculate_zone_based_position(good_new, roi_coords):
    """
    Menghitung posisi Y menggunakan pembagian zona dan median
    1. Membagi ROI menjadi 3 zona vertikal (atas, tengah, bawah)
    2. Memberikan bobot berbeda untuk tiap zona
    3. Menggunakan median untuk tiap zona
    """
    left_x, top_y, right_x, bottom_y = roi_coords
    roi_height = bottom_y - top_y
    zone_height = roi_height // 3
    
    # Membagi titik-titik ke dalam 3 zona
    top_zone = []
    middle_zone = []
    bottom_zone = []
    
    for point in good_new:
        x, y = point.ravel()
        relative_y = y - top_y  # posisi y relatif terhadap ROI
        
        if relative_y < zone_height:  # zona atas
            top_zone.append(y)
        elif relative_y < 2 * zone_height:  # zona tengah
            middle_zone.append(y)
        else:  # zona bawah
            bottom_zone.append(y)
    
    # Hitung median untuk tiap zona
    zone_medians = []
    if top_zone:
        zone_medians.append((np.median(top_zone), 0.2))  # bobot 0.2 untuk zona atas
    if middle_zone:
        zone_medians.append((np.median(middle_zone), 0.5))  # bobot 0.5 untuk zona tengah
    if bottom_zone:
        zone_medians.append((np.median(bottom_zone), 0.3))  # bobot 0.3 untuk zona bawah
    
    if not zone_medians:
        return None
    
    """
        Contoh perhitungan lengkap:
        1. Median tiap zona:
        - Zona Atas: 101
        - Zona Tengah: 150
        - Zona Bawah: 200

        2. Perhitungan weighted average:
        weighted_sum = (101 * 0.2) + (150 * 0.5) + (200 * 0.3)
                        = 20.2 + 75 + 60
                        = 155.2

        total_weight = 0.2 + 0.5 + 0.3 = 1.0

        final_position = 155.2 / 1.0 = 155.2

        Jadi posisi Y akhir yang digunakan adalah 155.2 pixels
    """
    
    # Hitung weighted position
    total_weight = sum(weight for _, weight in zone_medians)
    weighted_sum = sum(median * weight for median, weight in zone_medians)
    
    return weighted_sum / total_weight if total_weight > 0 else None

def draw_zones_and_points(frame, roi_coords, good_new):
    """
    Menggambar zona dan titik-titik dengan warna berbeda
    """
    left_x, top_y, right_x, bottom_y = roi_coords
    roi_height = bottom_y - top_y
    zone_height = roi_height // 3

    # Buat overlay untuk zona dengan transparansi
    overlay = frame.copy()
    
    # Gambar zona dengan warna berbeda (BGR format)
    # Zona atas (merah muda transparan)
    cv2.rectangle(overlay, 
                 (left_x, top_y), 
                 (right_x, top_y + zone_height),
                 (147, 20, 255), -1)  # Pink
    
    # Zona tengah (hijau transparan)
    cv2.rectangle(overlay, 
                 (left_x, top_y + zone_height), 
                 (right_x, top_y + 2*zone_height),
                 (0, 255, 0), -1)  # Green
    
    # Zona bawah (biru transparan)
    cv2.rectangle(overlay, 
                 (left_x, top_y + 2*zone_height), 
                 (right_x, bottom_y),
                 (255, 191, 0), -1)  # Blue
    
    # Aplikasikan transparansi
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Gambar titik-titik dengan warna sesuai zonanya
    for point in good_new:
        x, y = point.ravel()
        relative_y = y - top_y
        
        if relative_y < zone_height:  # zona atas
            color = (147, 20, 255)  # Pink
        elif relative_y < 2 * zone_height:  # zona tengah
            color = (0, 255, 0)  # Green
        else:  # zona bawah
            color = (255, 191, 0)  # Blue
            
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    
    # Tambahkan label zona dan bobot
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Top Zone (0.2)", (right_x + 10, top_y + zone_height//2), 
                font, 0.5, (147, 20, 255), 2)
    cv2.putText(frame, "Middle Zone (0.5)", (right_x + 10, top_y + 3*zone_height//2), 
                font, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Bottom Zone (0.3)", (right_x + 10, top_y + 5*zone_height//2), 
                font, 0.5, (255, 191, 0), 2)
    
    return frame

def process_webcam(model_path, max_seconds, x_size, y_size, shift_x, shift_y):
    """
    Memproses webcam untuk melacak gerakan bahu.
    Menggunakan optical flow dan pose detection untuk tracking.
    """
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_seconds)
    
    # Setup pose landmarker
    pose_landmarker = setup_pose_landmarker(model_path)
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=(4, 3))
    timestamps = []
    y_positions = []
    
    # Read first frame and get ROI
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame!")
    
    try:
        # Mendapatkan ROI awal dari frame pertama
        roi_coords = get_initial_roi(first_frame, model_path, 
                                   x_size=x_size, y_size=y_size,
                                   shift_x=shift_x, shift_y=shift_y)
        left_x, top_y, right_x, bottom_y = roi_coords
        
        # Inisialisasi tracking dengan Optical Flow
        old_frame = first_frame.copy()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize ROI and feature detection
        roi = old_gray[top_y:bottom_y, left_x:right_x]
        features = cv2.goodFeaturesToTrack(roi, 
                                         maxCorners=60,
                                         qualityLevel=0.15,
                                         minDistance=3,
                                         blockSize=7)
        
        if features is None:
            raise ValueError("No features found to track!")
            
        # Menyesuaikan koordinat fitur ke frame penuh
        features = np.float32(features)
        features[:,:,0] += left_x
        features[:,:,1] += top_y
        
        # LK params for better tracking
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if len(features) > 10:  # Ensure we have enough features
                # Calculate optical flow
                new_features, status, error = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, features, None, **lk_params)
                
                # Select good points
                good_old = features[status == 1]
                good_new = new_features[status == 1]
                
                # Draw tracks and calculate movement
                mask = np.zeros_like(frame)
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)
                
                frame = cv2.add(frame, mask)
                
                # Update tracking points
                if len(good_new) > 0:
                    # Ganti mean dengan zone-based position
                    y_pos = calculate_zone_based_position(good_new, (left_x, top_y, right_x, bottom_y))
                    if y_pos is not None:
                        current_time = time.time() - start_time
                        y_positions.append(y_pos)
                        timestamps.append(current_time)
                        features = good_new.reshape(-1, 1, 2)
                    
                    # Gambar zona dan titik-titik
                    frame = draw_zones_and_points(frame, (left_x, top_y, right_x, bottom_y), good_new)
                    
                    # Update plot
                    ax.clear()
                    ax.set_facecolor('none')
                    ax.patch.set_alpha(0.7)
                    ax.plot(timestamps, y_positions, 'g-', linewidth=2)
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Y Position (pixels)')
                    ax.set_title('Shoulder Movement')
                    ax.grid(True, alpha=0.3)

                    """
                    Contoh data yang terkumpul setelah beberapa frame:
                    
                    Frame    Timestamp    Y Position
                    1       0.1          155.2      <- bahu posisi normal
                    2       0.2          157.1      <- bahu bergerak turun
                    3       0.3          154.8      <- bahu bergerak naik
                    4       0.4          156.3      <- bahu turun lagi
                    
                    Pola ini menunjukkan gerakan naik-turun bahu saat bernapas
                    """
                    
                    # Set consistent axis limits
                    if len(timestamps) > 1:
                        ax.set_xlim(0, max_seconds)
                        y_min, y_max = min(y_positions), max(y_positions)
                        y_range = y_max - y_min
                        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
                
                    # Convert plot to image
                    fig.canvas.draw()
                    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    
                    # Resize plot
                    plot_height = int(height * 0.3)
                    plot_width = int(width * 0.3)
                    plot_img = cv2.resize(plot_img, (plot_width, plot_height))
                    
                    # Create mask for plot background
                    plot_gray = cv2.cvtColor(plot_img, cv2.COLOR_RGB2GRAY)
                    _, mask = cv2.threshold(plot_gray, 0, 255, cv2.THRESH_BINARY)
                    mask = mask.astype(bool)
                    
                    # Overlay plot on frame
                    padding = 20
                    y_offset = padding
                    x_offset = width - plot_width - padding
                    roi = frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width]
                    roi[mask] = plot_img[mask]
                    frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width] = roi
            else:
                # If we lose too many features, detect new ones
                roi = frame_gray[top_y:bottom_y, left_x:right_x]
                features = cv2.goodFeaturesToTrack(roi, 
                                                 maxCorners=100,
                                                 qualityLevel=0.01,
                                                 minDistance=7,
                                                 blockSize=7)
                if features is not None:
                    features = features + np.array([[left_x, top_y]], dtype=np.float32)
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("Shoulder Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Update for next frame
            old_gray = frame_gray.copy()
            frame_count += 1
            
    finally:
        plt.close(fig)
        cap.release()
        cv2.destroyAllWindows()
        
    return timestamps, y_positions

