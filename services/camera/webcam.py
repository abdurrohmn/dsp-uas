import cv2

class Webcam:
    def __init__(self, source=0, fps=30, resolution=(640, 480)):
        """
        Inisialisasi webcam.

        Parameters:
        - source (int or str): Sumber kamera (default: 0 untuk webcam default).
        - fps (int): Frame per second.
        - resolution (tuple): Resolusi kamera (width, height).
        """
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.fps = fps

    def read(self):
        """
        Membaca frame dari webcam.

        Returns:
        - ret (bool): Status pembacaan frame.
        - frame (numpy.ndarray): Frame yang dibaca.
        """
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """
        Melepaskan webcam.
        """
        self.cap.release()