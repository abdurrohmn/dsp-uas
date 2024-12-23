# plotting_signal_respiration.py

import matplotlib.pyplot as plt

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
