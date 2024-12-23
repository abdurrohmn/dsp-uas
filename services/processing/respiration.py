# respiration.py

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils_respiration import enhance_roi, get_initial_roi, prepare_plot, download_model, check_gpu
from chest_detection.py import initialize_pose_landmarker
from plotting_signal_respiration import plot_shoulder_movement

import mediapipe as mp

def process_video(landmarker, video_path, max_seconds=20, x_size=300, y_size=250, shift_x=0, shift_y=0):
    """
    Memproses video untuk melacak gerakan bahu.
    Menggunakan optical flow dan pose detection untuk tracking.
    
    Args:
        landmarker: Model pose detector
        video_path: Path ke file video
        max_seconds: Durasi maksimum video yang diproses
        x_size: Lebar ROI
        y_size: Tinggi ROI
        shift_x: Pergeseran horizontal ROI
        shift_y: Pergeseran vertikal ROI
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_seconds)
    
    # Initialize video writer with original frame size
    output_path = 'data/toby-shoulder-track.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Prepare plot
    fig, ax = prepare_plot()
    timestamps = []
    y_positions = []
    
    # Read first frame and get ROI
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame!")
    
    try:
        # Mendapatkan ROI awal dari frame pertama
        roi_coords = get_initial_roi(first_frame, landmarker, 
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
        
        # Adjust coordinates to full frame
        features[:,:,0] += left_x
        features[:,:,1] += top_y
        
        # LK params for better tracking
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        frame_count = 0
        pbar = tqdm(total=max_frames, desc='Processing frames')
        
        # Loop utama pemrosesan video
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
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
                    avg_y = np.mean(good_new[:, 1])
                    y_positions.append(avg_y)
                    timestamps.append(frame_count / fps)
                    features = good_new.reshape(-1, 1, 2)
                    
                    # Update plot
                    ax.clear()
                    ax.set_facecolor('none')
                    ax.patch.set_alpha(0.7)
                    ax.plot(timestamps, y_positions, 'g-', linewidth=2)
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Y Position (pixels)')
                    ax.set_title('Shoulder Movement')
                    ax.grid(True, alpha=0.3)
                    
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
                    
                    # Resize plot to desired size (e.g., 30% of the frame width and height)
                    plot_height = int(height * 0.3)  # 30% of frame height
                    plot_width = int(width * 0.3)    # 30% of frame width
                    plot_img = cv2.resize(plot_img, (plot_width, plot_height))
                    
                    # Create mask for plot background
                    plot_gray = cv2.cvtColor(plot_img, cv2.COLOR_RGB2GRAY)
                    _, mask_plot = cv2.threshold(plot_gray, 0, 255, cv2.THRESH_BINARY)
                    mask_plot = mask_plot.astype(bool)
                    
                    # Calculate position for overlay (top right corner with padding)
                    padding = 20
                    y_offset = padding
                    x_offset = width - plot_width - padding
                    
                    # Overlay plot on frame
                    roi_plot = frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width]
                    roi_plot[mask_plot] = plot_img[mask_plot]
                    frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width] = roi_plot
                    
                    # Write frame
                    out.write(frame)
            else:
                # If we lose too many features, detect new ones
                roi_new = frame_gray[top_y:bottom_y, left_x:right_x]
                features = cv2.goodFeaturesToTrack(roi_new, 
                                                 maxCorners=100,
                                                 qualityLevel=0.01,
                                                 minDistance=7,
                                                 blockSize=7)
                if features is not None:
                    features = features + np.array([[left_x, top_y]], dtype=np.float32)
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 2)
            
            # Update for next frame
            old_gray = frame_gray.copy()
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        plt.close(fig)
        cap.release()
        out.release()
        
        return timestamps, y_positions
        
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        cap.release()
        out.release()
        raise

def main():
    """
    Fungsi utama program.
    Menginisialisasi model dan memproses video untuk tracking gerakan bahu.
    """
    detector_image = None
    try:
        # 1. Download the model
        model_path = download_model()
        
        # 2. Prepare the pose landmarkers
        gpu_check = check_gpu()
        
        # Initialize PoseLandmarker
        landmarker = initialize_pose_landmarker(model_path, delegate=gpu_check)
        
        video_path = 'data/toby-rgb.mp4'
        print("\nProcessing video...")
        
        # Process video
        timestamps, y_positions = process_video(landmarker, video_path,
                                             max_seconds=20,
                                             x_size=300,
                                             y_size=200,
                                             shift_x=0,
                                             shift_y=100)
        
        print("Video processing completed!")
        
        # Optionally, plot the shoulder movement
        # plot_shoulder_movement(timestamps, y_positions)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    finally:
        if detector_image:
            detector_image.close()

if __name__ == "__main__":
    main()
