import matplotlib.pyplot as plt

class SignalPlotter:
    @staticmethod
    def plot_rgb_signals(r_signal, g_signal, b_signal):
        fig, axs = plt.subplots(3, 1, figsize=(20, 10))
        axs[0].plot(r_signal, color='red')
        axs[0].set_title('Sinyal Merah')
        axs[1].plot(g_signal, color='green')
        axs[1].set_title('Sinyal Hijau')
        axs[2].plot(b_signal, color='blue')
        axs[2].set_title('Sinyal Biru')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_rppg_comparison(raw_signal, filtered_signal):
        fig, axs = plt.subplots(2, 1, figsize=(20, 6))
        axs[0].plot(raw_signal, color='black')
        axs[0].set_title('Sinyal rPPG - Sebelum Pemfilteran')
        axs[1].plot(filtered_signal, color='black')
        axs[1].set_title('Sinyal rPPG - Setelah Pemfilteran')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_heart_rate(signal, peaks, heart_rate):
        plt.figure(figsize=(20, 5))
        plt.plot(signal, color='black')
        plt.plot(peaks, signal[peaks], 'x', color='red')
        plt.title(f'Detak Jantung: {heart_rate:.2f} BPM')
        plt.tight_layout()
        return plt.gcf()
