# ═══════════════════════════════════════════════════════════════
# inference_live.py
# Carga el modelo GAT entrenado y predice gestos en tiempo real
# capturando la pantalla. Muestra el gesto y la confianza.
# ═══════════════════════════════════════════════════════════════

import cv2                # captura, conversión y visualización
import torch              # cargar y ejecutar el modelo
import numpy as np        # manipulación de arrays
import mss                # captura de pantalla en tiempo real
from utils import HandLandmarkExtractor, normalize_landmarks, build_hand_graph
# HandLandmarkExtractor → detecta los 21 landmarks con MediaPipe
# normalize_landmarks   → centra en muñeca y escala a [-1, 1]
# build_hand_graph      → aquí sí se necesita — convierte landmarks en grafo

from train_gat_model import GestureGAT
# Reutilizamos la arquitectura definida en train_gat_model.py
# Principio DRY — no redefinimos lo que ya existe


# ── CONFIGURACIÓN ─────────────────────────────────────────────────
# Los mismos diccionarios que en data_collector — consistencia
GESTURES = {0: 'Piedra', 1: 'Papel', 2: 'Tijera'}
COLORS   = {0: (60, 60, 220), 1: (60, 180, 60), 2: (220, 160, 0)}


# ── CAPÍTULO 1: Cargar el modelo entrenado ───────────────────────
def load_model():
    # Crea la arquitectura vacía — misma estructura que al entrenar
    # Sin esto, load_state_dict no sabe dónde poner los pesos
    model = GestureGAT(num_classes=3)

    # Carga los pesos entrenados desde el archivo .pt
    # state_dict es un diccionario {nombre_capa: tensor_de_pesos}
    model.load_state_dict(torch.load('gesture_gat.pt'))

    # Mueve todos los pesos a la GPU (RTX 3060)
    model = model.cuda()

    # Modo evaluación — desactiva el dropout
    # En entrenamiento el dropout apaga neuronas aleatoriamente
    # En inferencia queremos predicciones deterministas y consistentes
    model.eval()

    return model


# ── CAPÍTULO 2: El programa principal ────────────────────────────
def main():
    # Carga el modelo con pesos entrenados
    model = load_model()

    # Inicializa MediaPipe — una sola vez para todo el programa
    extractor  = HandLandmarkExtractor()

    # El grafo de conexiones es siempre el mismo — se calcula una vez
    edge_index = build_hand_graph()

    # Utilidades de MediaPipe para dibujar los 21 puntos
    mp_draw  = __import__('mediapipe').solutions.drawing_utils
    mp_hands = __import__('mediapipe').solutions.hands

    with mss.MSS() as sct:
        monitor = sct.monitors[1]  # monitor principal

        # ── Loop principal — corre ~30 veces por segundo ──
        while True:
            # 1. Captura un frame de la pantalla
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)

            # 2. Convierte BGRA → BGR (MSS captura con canal alpha)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 3. Extrae landmarks — None si no hay mano
            landmarks = extractor.extract(frame)

            if landmarks is not None:
                # 4. Dibuja los 21 puntos sobre el frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = extractor.hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, hand_lm,
                            mp_hands.HAND_CONNECTIONS
                        )

                # 5. Prepara el grafo para el modelo
                normalized = normalize_landmarks(landmarks)

                # Convierte a tensor y mueve a GPU
                x  = torch.tensor(normalized, dtype=torch.float).cuda()
                ei = edge_index.cuda()

                # Empaqueta en objeto Data — la "caja" de PyTorch Geometric
                from torch_geometric.data import Data, Batch
                data  = Data(x=x, edge_index=ei)

                # Batch.from_data_list agrupa el grafo en un batch de 1
                # El modelo espera batches, no grafos individuales
                batch = Batch.from_data_list([data])

                # 6. Predicción — sin calcular gradientes (solo inferencia)
                with torch.no_grad():
                    out  = model(batch)

                    # argmax → índice del valor más alto = gesto predicho
                    pred = out.argmax(dim=1).item()

                    # softmax convierte logits en probabilidades [0, 1]
                    # que suman 1.0 — .max() toma la más alta = confianza
                    conf = torch.softmax(out, dim=1).max().item()

                # 7. Muestra el gesto y confianza en pantalla
                gesture = GESTURES[pred]
                color   = COLORS[pred]

                # f'{conf:.0%}' formatea 0.85 como "85%"
                cv2.putText(frame, f'{gesture} ({conf:.0%})', (30, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

            else:
                # Si no hay mano, avisa al usuario
                cv2.putText(frame, 'Sin mano', (30, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)

            # 8. Muestra el frame en ventana
            cv2.imshow('GAT Inference', frame)

            # 9. Sale con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


# Solo ejecuta si se corre directamente — no si se importa
if __name__ == '__main__':
    main()