import mediapipe as mp
import numpy as np

class HandLandmarkExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, #no es imagen estática, es video en tiempo real. Activa optimizaciones de rendimiento
            max_num_hands=1, #define el numero de manos. Más manos, más procesamiento.
            min_detection_confidence=0.7 #sólo reporta landmarks si está 70% seguro de haber detectado una mano
        )