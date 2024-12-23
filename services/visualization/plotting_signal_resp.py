# src/plotting.py

import matplotlib.pyplot as plt
import cv2
import matplotlib
import numpy as np

class SignalPlotterResp:
# Use 'Agg' backend for environments without a display (e.g., servers)
    matplotlib.use('Agg')

def overlay_plot_on_frame(frame, timestamps, y_positions):
    if not timestamps or not y_positions:  # Skip if no data
        return frame

    # Use Agg backend which doesn't require a GUI
    plt.switch_backend('Agg')
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(timestamps, y_positions, "g-", linewidth=2)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title("Shoulder Movement")

    # Convert plot to image
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Clean up
    plt.close(fig)
    
    # Resize plot_img to fit in a corner of the frame
    h, w = frame.shape[:2]
    plot_h, plot_w = plot_img.shape[:2]
    scale = min(h/3/plot_h, w/3/plot_w)  # Make plot 1/3 of frame size
    new_h, new_w = int(plot_h*scale), int(plot_w*scale)
    plot_img = cv2.resize(plot_img, (new_w, new_h))
    
    # Overlay plot in top-right corner
    frame[10:10+new_h, w-new_w-10:w-10] = plot_img
    
    return frame

def plot_shoulder_movement(timestamps, y_positions):
        """
        Membuat plot gerakan bahu terhadap waktu.
    
        Args:
            timestamps: Array waktu
            y_positions: Array posisi y bahu
        """
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, y_positions, label='Average Y Position', color='green')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Chest/Shoulder Movement Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()