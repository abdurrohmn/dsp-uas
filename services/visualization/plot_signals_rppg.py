import matplotlib.pyplot as plt
import numpy as np
import cv2

class SignalPlotter:
    @staticmethod
    def plot_rgb_signals(r_signal, g_signal, b_signal):
        fig, axs = plt.subplots(3, 1, figsize=(15, 8))
        
        # Plot Red signal
        axs[0].plot(r_signal, color='red', linewidth=1.5)
        axs[0].set_title('Sinyal Merah', fontsize=12, pad=10)
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].set_ylabel('Amplitudo', fontsize=10)
        
        # Plot Green signal
        axs[1].plot(g_signal, color='green', linewidth=1.5)
        axs[1].set_title('Sinyal Hijau', fontsize=12, pad=10)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].set_ylabel('Amplitudo', fontsize=10)
        
        # Plot Blue signal
        axs[2].plot(b_signal, color='blue', linewidth=1.5)
        axs[2].set_title('Sinyal Biru', fontsize=12, pad=10)
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].set_ylabel('Amplitudo', fontsize=10)
        axs[2].set_xlabel('Sampel', fontsize=10)
        
        plt.tight_layout(pad=2.0)
        return fig

    @staticmethod
    def plot_heart_rate(signal, peaks, heart_rate):
        fig = plt.figure(figsize=(15, 4))
        plt.plot(signal, color='blue', linewidth=1.5, label='Sinyal')
        plt.plot(peaks, signal[peaks], 'rx', markersize=8, label='Puncak')
        plt.title(f'Detak Jantung: {heart_rate:.2f} BPM', fontsize=12, pad=10)
        plt.xlabel('Sampel', fontsize=10)
        plt.ylabel('Amplitudo', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout(pad=1.5)
        return fig
