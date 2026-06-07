import mediapipe as mp
import numpy as np
import cv2
import torch

class HandLandmarkExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, #no es imagen estática, es video en tiempo real. Activa optimizaciones de rendimiento
            max_num_hands=1, #define el numero de manos. Más manos, más procesamiento.
            min_detection_confidence=0.7 #sólo reporta landmarks si está 70% seguro de haber detectado una mano
        )
    def extract(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Convierte el frame de BGR (OpenCV) a RGB (Mediapipe)
        results = self.hands.process(frame_rgb) #MediaPipe analiza la imagen y busca manos. results contiene todo lo que encontró.
        
        if results.multi_hand_landmarks: #Pregunta: ¿encontró al menos una mano? Si no hay mano en el frame, multi_hand_landmarks es None y saltamos al return None.
            landmarks = results.multi_hand_landmarks[0] #Toma la primera mano detectada — ( configuramos max_num_hands=1)
            return np.array([[lm.x, lm.y, lm.z] #Recorre los 21 landmarks y construye un array con forma [21, 3]
                            for lm in landmarks.landmark])
        return None

def normalize_landmarks(landmarks):
    base = landmarks[0].copy() #la muñeca
    landmarks = landmarks - base #Resta la muñeca a todos los 21 puntos para dejar la muñeca en el centro (0, 0, 0)
    max_val = np.max(np.abs(landmarks)) #Encuentra el valor absoluto más grande entre todos los 63 números.
    if max_val > 0:
        landmarks = landmarks / max_val #Divide todos los puntos por ese valor máximo. Resultado: todos los números quedan entre -1 y 1. Esto resuelve el problema de distancia — mano cerca o lejos, siempre el mismo rango.
    return landmarks #Devuelve el array normalizado con forma [21, 3] — mismo shape que antes, pero valores transformados.

def build_hand_graph(landmarks):
    edges = [
        (0,1),(1,2),(2,3),(3,4),        # pulgar
        (0,5),(5,6),(6,7),(7,8),         # índice
        (0,9),(9,10),(10,11),(11,12),    # medio
        (0,13),(13,14),(14,15),(15,16),  # anular
        (0,17),(17,18),(18,19),(19,20),  # meñique
        (5,9),(9,13),(13,17)             # nudillos
    ]
    edge_index = torch.tensor(
        [[e[0] for e in edges] + [e[1] for e in edges],
         [e[1] for e in edges] + [e[0] for e in edges]],
        dtype=torch.long
    )
    return edge_index