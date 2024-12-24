# modules/detection/face_detection.py

import mediapipe as mp
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        """
        Inisialisasi detektor wajah menggunakan MediaPipe.

        Parameter:
        - model_selection (int): Pemilihan model MediaPipe Face Detection.
        - min_detection_confidence (float): Ambang batas kepercayaan untuk deteksi.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection, min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.prev_bbox = None
        self.smoothing_factor = 0.7  # For bbox smoothing

    def detect(self, frame):
        """
        Deteksi wajah pada frame.

        Parameter:
        - frame (numpy.ndarray): Frame input dalam format BGR.

        Return:
        - detections (list): Daftar deteksi wajah.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        return results.detections if results.detections else []

    def get_bounding_box(self, detection, frame_shape):
        """
        Mendapatkan koordinat bounding box dari deteksi wajah dengan penyesuaian untuk forehead.

        Parameter:
        - detection: Objek deteksi wajah dari MediaPipe.
        - frame_shape (tuple): Dimensi dari frame (tinggi, lebar, channels).

        Return:
        - x (int): Koordinat x dari bounding box.
        - y (int): Koordinat y dari bounding box.
        - width (int): Lebar bounding box.
        - height (int): Tinggi bounding box.
        """
        bbox = detection.location_data.relative_bounding_box
        h, w = frame_shape[:2]
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Extend the box upward to include forehead
        forehead_extension = int(height * 0.3)  # Extend box upward by 30% of face height
        y = max(0, y - forehead_extension)
        height += forehead_extension
        
        return x, y, width, height

    def adjust_bbox(self, bbox, size_from_center=70):
        """
        Penyesuaian bounding box yang ditingkatkan dengan smoothing.

        Parameter:
        - bbox (tuple): Koordinat bounding box (x, y, width, height).
        - size_from_center (int): Ukuran dari pusat bounding box.

        Return:
        - new_x (int): Koordinat x yang telah disesuaikan.
        - new_y (int): Koordinat y yang telah disesuaikan.
        - new_width (int): Lebar bounding box yang telah disesuaikan.
        - new_height (int): Tinggi bounding box yang telah disesuaikan.
        """
        x, y, width, height = bbox
        bbox_center_x = x + width // 2
        bbox_center_y = y + height // 2

        # Apply size adjustment
        new_x = bbox_center_x - size_from_center
        new_y = bbox_center_y - size_from_center
        new_width = size_from_center * 2
        new_height = size_from_center * 2

        # Smooth bbox transitions
        if self.prev_bbox is not None:
            prev_x, prev_y, prev_w, prev_h = self.prev_bbox
            new_x = int(self.smoothing_factor * new_x + (1 - self.smoothing_factor) * prev_x)
            new_y = int(self.smoothing_factor * new_y + (1 - self.smoothing_factor) * prev_y)
            new_width = int(self.smoothing_factor * new_width + (1 - self.smoothing_factor) * prev_w)
            new_height = int(self.smoothing_factor * new_height + (1 - self.smoothing_factor) * prev_h)

        self.prev_bbox = (new_x, new_y, new_width, new_height)
        return new_x, new_y, new_width, new_height

    def get_forehead_roi(self, frame, bbox):
        """
        Ekstrak ROI forehead dengan penyesuaian area.

        Parameter:
        - frame (numpy.ndarray): Frame input dalam format BGR.
        - bbox (tuple): Koordinat bounding box (x, y, width, height).

        Return:
        - forehead_roi (numpy.ndarray): ROI dari forehead.
        """
        x, y, w, h = bbox
        forehead_height = int(h * 0.25)  # Take top 25% of extended face area
        forehead_start = y
        forehead_end = y + forehead_height
        
        # pastikan ROI berada dalam batas frame
        forehead_start = max(0, forehead_start)
        forehead_end = min(frame.shape[0], forehead_end)
        
        forehead_roi = frame[forehead_start:forehead_end, x:x+w]
        return forehead_roi
