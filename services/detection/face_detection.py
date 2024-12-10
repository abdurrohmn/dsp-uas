# modules/detection/face_detection.py

import mediapipe as mp
import cv2

class FaceDetector:
    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        """
        Inisialisasi detektor wajah menggunakan MediaPipe.

        Parameters:
        - model_selection (int): Model selection untuk MediaPipe Face Detection.
        - min_detection_confidence (float): Ambang batas kepercayaan untuk deteksi.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection, min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame):
        """
        Deteksi wajah pada frame.

        Parameters:
        - frame (numpy.ndarray): Frame input dalam format BGR.

        Returns:
        - detections (list): Daftar deteksi wajah.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        return results.detections if results.detections else []

    def get_bounding_box(self, detection, frame_shape):
        """
        Mendapatkan koordinat bounding box dari deteksi wajah.

        Parameters:
        - detection: Objek deteksi wajah dari MediaPipe.
        - frame_shape (tuple): Shape dari frame (height, width, channels).

        Returns:
        - x, y, width, height (int): Koordinat bounding box dalam piksel.
        """
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame_shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        return x, y, width, height

    def adjust_bbox(self, bbox, size_from_center=70):
        """
        Menyesuaikan bounding box agar memiliki ukuran tertentu dari pusat.

        Parameters:
        - bbox (tuple): Koordinat bounding box (x, y, width, height).
        - size_from_center (int): Ukuran dari pusat bounding box.

        Returns:
        - new_x, new_y, new_width, new_height (int): Koordinat bounding box yang disesuaikan.
        """
        x, y, width, height = bbox
        bbox_center_x = x + width // 2
        bbox_center_y = y + height // 2
        new_x = bbox_center_x - size_from_center
        new_y = bbox_center_y - size_from_center
        new_width = size_from_center * 2
        new_height = size_from_center * 2
        return new_x, new_y, new_width, new_height
