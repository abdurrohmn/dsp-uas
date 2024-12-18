import numpy as np
import scipy.signal as signal

class RPPG:
    def __init__(self, fps=30):
        self.fps = fps

    def process_pos(self, rgb_signals):
        """Implementasi algoritma POS"""
        rgb_signals = np.array(rgb_signals).reshape(1, 3, -1)
        rppg_signal = self.cpu_POS(rgb_signals, fps=self.fps)
        return rppg_signal.reshape(-1)

    def filter_signal(self, rppg_signal, lowcut=0.9, highcut=2.4, order=3):
        """Menerapkan filter bandpass ke sinyal rPPG"""
        b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=self.fps)
        filtered_signal = signal.filtfilt(b, a, rppg_signal)
        return filtered_signal

    def compute_heart_rate(self, filtered_signal, prominence=0.5):
        """Menghitung detak jantung dari sinyal rPPG yang telah difilter"""
        normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
        peaks, _ = signal.find_peaks(x=normalized_signal, prominence=prominence)
        heart_rate = 60 * len(peaks) / (len(filtered_signal) / self.fps)
        return heart_rate, peaks, normalized_signal

    def cpu_POS(self, signal, **kargs):
        """
        POS method on CPU using Numpy.

        The dictionary parameters are: {'fps':float}.

        Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
        """
        # Run the pos algorithm on the RGB color signal c with sliding window length wlen
        # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
        eps = 10**-9
        X = signal
        e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
        w = int(1.6 * kargs['fps'])   # window length

        # stack e times fixed mat P
        P = np.array([[0, 1, -1], [-2, 1, 1]])
        Q = np.stack([P for _ in range(e)], axis=0)

        # Initialize (1)
        H = np.zeros((e, f))
        for n in np.arange(w, f):
            # Start index of sliding window (4)
            m = n - w + 1
            # Temporal normalization (5)
            Cn = X[:, :, m:(n + 1)]
            M = 1.0 / (np.mean(Cn, axis=2)+eps)
            M = np.expand_dims(M, axis=2)  # shape [e, c, w]
            Cn = np.multiply(M, Cn)

            # Projection (6)
            S = np.dot(Q, Cn)
            S = S[0, :, :, :]
            S = np.swapaxes(S, 0, 1)    # remove 3-th dim

            # Tuning (7)
            S1 = S[:, 0, :]
            S2 = S[:, 1, :]
            alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
            alpha = np.expand_dims(alpha, axis=1)
            Hn = np.add(S1, alpha * S2)
            Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
            # Overlap-adding (8)
            H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

        return H
