import mediapipe as mp
import numpy as np
import cv2

class HandLandmarkExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, #no es imagen estática, es video en tiempo real. Activa optimizaciones de rendimiento
            max_num_hands=1, #define el numero de manos. Más manos, más procesamiento.
            min_detection_confidence=0.7 #sólo reporta landmarks si está 70% seguro de haber detectado una mano
        )
    def extract(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] 
                            for lm in landmarks.landmark])
        return None