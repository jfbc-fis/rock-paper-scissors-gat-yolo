import cv2
import torch
import numpy as np
import mss
from utils import HandLandmarkExtractor, normalize_landmarks, build_hand_graph
from train_gat_model import GestureGAT

GESTURES = {0: 'Piedra', 1: 'Papel', 2: 'Tijera'}
COLORS   = {0: (60, 60, 220), 1: (60, 180, 60), 2: (220, 160, 0)}

def load_model():
    # Crea la arquitectura vacía
    model = GestureGAT(num_classes=3)
    # Carga los pesos entrenados
    model.load_state_dict(torch.load('gesture_gat.pt'))
    # Mueve a GPU
    model = model.cuda()
    # Modo evaluación — desactiva dropout
    model.eval()
    return model

def main():
    model     = load_model()
    extractor = HandLandmarkExtractor()
    edge_index = build_hand_graph()
    mp_draw   = __import__('mediapipe').solutions.drawing_utils
    mp_hands  = __import__('mediapipe').solutions.hands

    with mss.MSS() as sct:
        monitor = sct.monitors[1]

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            landmarks = extractor.extract(frame)

            if landmarks is not None:
                # Dibuja landmarks
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = extractor.hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, hand_lm,
                            mp_hands.HAND_CONNECTIONS
                        )

                # Prepara el grafo para el modelo
                normalized  = normalize_landmarks(landmarks)
                x           = torch.tensor(normalized, dtype=torch.float).cuda()
                ei          = edge_index.cuda()

                from torch_geometric.data import Data, Batch
                data        = Data(x=x, edge_index=ei)
                batch       = Batch.from_data_list([data])

                # Predicción
                with torch.no_grad():
                    out  = model(batch)
                    pred = out.argmax(dim=1).item()
                    conf = torch.softmax(out, dim=1).max().item()

                # Muestra resultado
                gesture = GESTURES[pred]
                color   = COLORS[pred]
                cv2.putText(frame, f'{gesture} ({conf:.0%})', (30, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

            else:
                cv2.putText(frame, 'Sin mano', (30, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)

            cv2.imshow('GAT Inference', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()