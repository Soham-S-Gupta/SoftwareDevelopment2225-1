from collections.abc import Sequence

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[list[float]], tuple[int, int, int, int] | None]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return [], None

        height, width = frame_bgr.shape[:2]
        flat_landmarks: list[list[float]] = []
        xs: list[int] = []
        ys: list[int] = []

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                flat_landmarks.append([landmark.x, landmark.y, landmark.z])
                xs.append(int(landmark.x * width))
                ys.append(int(landmark.y * height))

        x1 = max(min(xs) - 20, 0)
        y1 = max(min(ys) - 20, 0)
        x2 = min(max(xs) + 20, width)
        y2 = min(max(ys) + 20, height)
        return flat_landmarks, (x1, y1, x2, y2)

    @staticmethod
    def landmarks_to_vector(landmarks: Sequence[Sequence[float]], expected_points: int = 42) -> np.ndarray:
        vector = np.zeros((expected_points, 3), dtype=np.float32)
        for idx, point in enumerate(landmarks[:expected_points]):
            vector[idx] = np.asarray(point, dtype=np.float32)
        vector[:, 0] -= vector[:, 0].mean()
        vector[:, 1] -= vector[:, 1].mean()
        scale = np.max(np.abs(vector[:, :2]))
        if scale > 0:
            vector[:, :2] /= scale
        return vector.reshape(-1)
