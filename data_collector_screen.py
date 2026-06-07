import cv2
import csv
import numpy as np
import mss
import mediapipe as mp
from utils import HandLandmarkExtractor, normalize_landmarks

# ── Configuración global ──────────────────────────────────────────
GESTURES = {0: 'Piedra', 1: 'Papel', 2: 'Tijera'}
OUTPUT_FILE = 'hand_landmarks.csv'
SAMPLES_PER_GESTURE = 200

# Color BGR para cada gesto
COLORS = {
    0: (60, 60, 220),    # rojo  → piedra
    1: (60, 180, 60),    # verde → papel
    2: (220, 160, 0),    # azul  → tijera
    None: (180, 180, 180)  # gris → sin gesto
}

def draw_panel(frame, current_gesture, count, collected):
    """Dibuja el panel lateral derecho con toda la información."""
    h, w = frame.shape[:2]
    panel_w = 280

    # Fondo del panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    color = COLORS[current_gesture]
    x = w - panel_w + 16

    # ── Título ──
    cv2.putText(frame, 'RECOLECTOR', (x, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'de landmarks', (x, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.line(frame, (x, 75), (w - 16, 75), (80, 80, 80), 1)

    # ── Instrucciones ──
    cv2.putText(frame, 'Teclas:', (x, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, '0  Piedra', (x, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 1)
    cv2.putText(frame, '1  Papel', (x, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[1], 1)
    cv2.putText(frame, '2  Tijera', (x, 172),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[2], 1)
    cv2.putText(frame, 'q  Salir', (x, 194),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.line(frame, (x, 208), (w - 16, 208), (80, 80, 80), 1)

    # ── Progreso por gesto ──
    cv2.putText(frame, 'Progreso:', (x, 232),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    bar_w = panel_w - 32
    for gid, name in GESTURES.items():
        y_base = 252 + gid * 52
        done = collected[gid]
        pct = min(done / SAMPLES_PER_GESTURE, 1.0)
        filled = int(bar_w * pct)
        g_color = COLORS[gid]

        # Etiqueta con contador
        label = f'{name}  {done}/{SAMPLES_PER_GESTURE}'
        cv2.putText(frame, label, (x, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, g_color, 1)

        # Barra de fondo
        cv2.rectangle(frame, (x, y_base + 6),
                      (x + bar_w, y_base + 22), (70, 70, 70), -1)

        # Barra de progreso
        if filled > 0:
            cv2.rectangle(frame, (x, y_base + 6),
                          (x + filled, y_base + 22), g_color, -1)

        # Marca de completado
        if done >= SAMPLES_PER_GESTURE:
            cv2.putText(frame, 'OK', (x + bar_w - 28, y_base + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.line(frame, (x, 420), (w - 16, 420), (80, 80, 80), 1)

    # ── Estado actual ──
    if current_gesture is None:
        estado = 'Esperando tecla...'
        e_color = (160, 160, 160)
    elif count >= SAMPLES_PER_GESTURE:
        estado = f'{GESTURES[current_gesture]} completo!'
        e_color = (60, 220, 60)
    else:
        estado = f'Grabando {GESTURES[current_gesture]}...'
        e_color = color

    cv2.putText(frame, estado, (x, 448),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, e_color, 1)

    # ── Total recolectado ──
    total = sum(collected.values())
    total_needed = SAMPLES_PER_GESTURE * len(GESTURES)
    cv2.putText(frame, f'Total: {total}/{total_needed}', (x, 475),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def main():
    extractor = HandLandmarkExtractor()
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Cuántas muestras ya tenemos por gesto (lee el CSV si existe)
    collected = {0: 0, 1: 0, 2: 0}
    try:
        with open(OUTPUT_FILE, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    label = int(row[-1])
                    if label in collected:
                        collected[label] += 1
        print(f"CSV existente detectado: {collected}")
    except FileNotFoundError:
        print("CSV nuevo — empezando desde cero")

    with mss.MSS() as sct:
        monitor = sct.monitors[1]

        print("Recolector iniciado")
        print("Presiona 0=Piedra  1=Papel  2=Tijera  q=Salir")

        current_gesture = None
        count = 0

        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)

            while True:
                # Captura y convierte el frame
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Extrae landmarks
                landmarks = extractor.extract(frame)

                # Dibuja landmarks si detecta mano
                if landmarks is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = extractor.hands.process(frame_rgb)
                    if results.multi_hand_landmarks:
                        for hand_lm in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(
                                frame, hand_lm,
                                mp_hands.HAND_CONNECTIONS
                            )

                # Dibuja panel lateral
                frame = draw_panel(frame, current_gesture, count, collected)

                cv2.imshow('Recolector de Landmarks', frame)

                # Manejo de teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in (ord('0'), ord('1'), ord('2')):
                    current_gesture = int(chr(key))
                    count = collected[current_gesture]

                # Guarda muestra si hay mano y gesto activo
                if landmarks is not None and current_gesture is not None:
                    if count < SAMPLES_PER_GESTURE:
                        normalized = normalize_landmarks(landmarks)
                        row = normalized.flatten().tolist() + [current_gesture]
                        writer.writerow(row)
                        count += 1
                        collected[current_gesture] = count

        cv2.destroyAllWindows()
        print(f"\nRecoleccion finalizada: {collected}")


if __name__ == '__main__':
    main()