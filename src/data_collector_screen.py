# ═══════════════════════════════════════════════════════════════
# data_collector_screen.py
# Captura la pantalla en tiempo real, detecta la mano con
# MediaPipe y guarda los 21 landmarks normalizados en un CSV.
# Ese CSV es el dataset con el que después entrena el modelo GAT.
# ═══════════════════════════════════════════════════════════════

import cv2          # captura, conversión de colores y visualización
import csv          # lectura y escritura del archivo de datos
import numpy as np  # manipulación de arrays numéricos
import mss          # captura de región de pantalla (Teams/Zoom)
import mediapipe as mp  # utilidades para dibujar los landmarks
from utils import HandLandmarkExtractor, normalize_landmarks
# HandLandmarkExtractor → detecta y extrae los 21 puntos de la mano
# normalize_landmarks   → centra en muñeca y escala a [-1, 1]
# build_hand_graph NO se importa aquí — se necesita en el entrenamiento


# ── CONFIGURACIÓN GLOBAL ──────────────────────────────────────────
# Diccionario: número → nombre del gesto (para mostrar al usuario)
GESTURES = {0: 'Piedra', 1: 'Papel', 2: 'Tijera'}

# Nombre del archivo donde se guardan los landmarks recolectados
OUTPUT_FILE = 'hand_landmarks.csv'

# Cuántas muestras recolectar por cada gesto
# 200 × 3 gestos = 600 filas en el CSV
SAMPLES_PER_GESTURE = 200

# Colores en formato BGR (Blue, Green, Red — el orden de OpenCV)
# Cada gesto tiene su propio color para identificarlo visualmente
COLORS = {
    0: (60, 60, 220),      # rojo   → piedra
    1: (60, 180, 60),      # verde  → papel
    2: (220, 160, 0),      # azul   → tijera
    None: (180, 180, 180)  # gris   → sin gesto activo
}


# ── CAPÍTULO 1: El panel lateral ─────────────────────────────────
# Función independiente — recibe todo por parámetro, no necesita self
# Dibuja la interfaz visual sobre el frame capturado
def draw_panel(frame, current_gesture, count, collected):
    """Dibuja el panel lateral derecho con toda la información."""

    # Obtiene las dimensiones del frame actual
    h, w = frame.shape[:2]
    panel_w = 280  # ancho del panel lateral en píxeles

    # Crea un fondo semitransparente para el panel
    # overlay es una copia del frame — se oscurece y se mezcla con el original
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (30, 30, 30), -1)
    # addWeighted mezcla overlay (75%) con frame (25%) → efecto de transparencia
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Color del gesto activo — None devuelve gris si no hay gesto seleccionado
    color = COLORS[current_gesture]
    x = w - panel_w + 16  # margen izquierdo del panel

    # ── Título del panel ──
    cv2.putText(frame, 'RECOLECTOR', (x, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'de landmarks', (x, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    # Línea separadora horizontal
    cv2.line(frame, (x, 75), (w - 16, 75), (80, 80, 80), 1)

    # ── Instrucciones de teclas ──
    cv2.putText(frame, 'Teclas:', (x, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    # Cada gesto se muestra con su color específico
    cv2.putText(frame, '0  Piedra', (x, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 1)
    cv2.putText(frame, '1  Papel', (x, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[1], 1)
    cv2.putText(frame, '2  Tijera', (x, 172),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[2], 1)
    cv2.putText(frame, 'q  Salir', (x, 194),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.line(frame, (x, 208), (w - 16, 208), (80, 80, 80), 1)

    # ── Barras de progreso por gesto ──
    cv2.putText(frame, 'Progreso:', (x, 232),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    bar_w = panel_w - 32  # ancho disponible para la barra

    for gid, name in GESTURES.items():
        # Posición vertical de cada barra — separadas 52 píxeles
        y_base = 252 + gid * 52
        done = collected[gid]  # muestras recolectadas de este gesto

        # Calcula qué fracción de la barra llenar (máximo 1.0 = 100%)
        pct = min(done / SAMPLES_PER_GESTURE, 1.0)
        filled = int(bar_w * pct)  # píxeles a rellenar
        g_color = COLORS[gid]

        # Etiqueta: "Piedra  45/200"
        label = f'{name}  {done}/{SAMPLES_PER_GESTURE}'
        cv2.putText(frame, label, (x, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, g_color, 1)

        # Fondo gris de la barra (vacío)
        cv2.rectangle(frame, (x, y_base + 6),
                      (x + bar_w, y_base + 22), (70, 70, 70), -1)

        # Relleno de la barra según progreso
        if filled > 0:
            cv2.rectangle(frame, (x, y_base + 6),
                          (x + filled, y_base + 22), g_color, -1)

        # Marca "OK" cuando el gesto está completo
        if done >= SAMPLES_PER_GESTURE:
            cv2.putText(frame, 'OK', (x + bar_w - 28, y_base + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.line(frame, (x, 420), (w - 16, 420), (80, 80, 80), 1)

    # ── Estado actual del recolector ──
    if current_gesture is None:
        estado = 'Esperando tecla...'   # no se ha seleccionado gesto
        e_color = (160, 160, 160)
    elif count >= SAMPLES_PER_GESTURE:
        estado = f'{GESTURES[current_gesture]} completo!'  # gesto terminado
        e_color = (60, 220, 60)
    else:
        estado = f'Grabando {GESTURES[current_gesture]}...'  # grabando activamente
        e_color = color

    cv2.putText(frame, estado, (x, 448),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, e_color, 1)

    # ── Contador total ──
    total = sum(collected.values())
    total_needed = SAMPLES_PER_GESTURE * len(GESTURES)  # 600
    cv2.putText(frame, f'Total: {total}/{total_needed}', (x, 475),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame  # devuelve el frame con el panel dibujado encima


# ── CAPÍTULO 2: El programa principal ────────────────────────────
def main():
    # Crea el extractor de landmarks — inicializa MediaPipe una sola vez
    extractor = HandLandmarkExtractor()

    # Utilidades de MediaPipe para dibujar los 21 puntos sobre el frame
    mp_draw  = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # ── Lee el CSV existente para no perder muestras previas ──
    collected = {0: 0, 1: 0, 2: 0}
    try:
        with open(OUTPUT_FILE, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    # La última columna es la etiqueta del gesto
                    label = int(row[-1])
                    if label in collected:
                        collected[label] += 1
        print(f"CSV existente detectado: {collected}")
    except FileNotFoundError:
        # Si no existe el CSV, empieza desde cero
        print("CSV nuevo — empezando desde cero")

    # ── Captura de pantalla con MSS ──
    with mss.MSS() as sct:
        # monitors[1] es el monitor principal
        # monitors[0] sería todos los monitores combinados
        monitor = sct.monitors[1]

        print("Recolector iniciado")
        print("Presiona 0=Piedra  1=Papel  2=Tijera  q=Salir")

        current_gesture = None  # gesto activo seleccionado por el usuario
        count = 0               # muestras recolectadas del gesto actual

        # Abre el CSV en modo 'a' (append) — agrega sin borrar lo existente
        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)

            # ── Loop principal — corre ~30 veces por segundo ──
            while True:
                # 1. Captura un frame de la pantalla
                screenshot = sct.grab(monitor)

                # 2. Convierte a array de NumPy (formato que OpenCV entiende)
                frame = np.array(screenshot)

                # 3. MSS captura en BGRA (con canal alpha) → convertir a BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # 4. Extrae los 21 landmarks — devuelve None si no hay mano
                landmarks = extractor.extract(frame)

                # 5. Si detectó mano, dibuja los 21 puntos y conexiones
                if landmarks is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = extractor.hands.process(frame_rgb)
                    if results.multi_hand_landmarks:
                        for hand_lm in results.multi_hand_landmarks:
                            # draw_landmarks dibuja puntos y líneas de conexión
                            mp_draw.draw_landmarks(
                                frame, hand_lm,
                                mp_hands.HAND_CONNECTIONS
                            )

                # 6. Dibuja el panel lateral con instrucciones y progreso
                frame = draw_panel(frame, current_gesture, count, collected)

                # 7. Muestra el frame en ventana
                cv2.imshow('Recolector de Landmarks', frame)

                # 8. Espera 1ms por una tecla
                # & 0xFF garantiza compatibilidad en sistemas de 64 bits
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break  # sale del loop y termina el programa

                elif key in (ord('0'), ord('1'), ord('2')):
                    # Cambia el gesto activo y retoma el conteo desde donde estaba
                    current_gesture = int(chr(key))
                    count = collected[current_gesture]

                # 9. Si hay mano Y gesto activo Y aún faltan muestras → guarda
                if landmarks is not None and current_gesture is not None:
                    if count < SAMPLES_PER_GESTURE:
                        # Normaliza: centra en muñeca y escala a [-1, 1]
                        normalized = normalize_landmarks(landmarks)

                        # flatten() convierte [21, 3] → [63] números planos
                        # + [current_gesture] agrega la etiqueta al final
                        row = normalized.flatten().tolist() + [current_gesture]

                        # Escribe la fila en el CSV
                        writer.writerow(row)
                        count += 1
                        collected[current_gesture] = count

        # Cierra todas las ventanas de OpenCV al salir
        cv2.destroyAllWindows()
        print(f"\nRecoleccion finalizada: {collected}")


# Solo ejecuta main() si este archivo se corre directamente
# Si otro archivo lo importa, NO ejecuta nada automáticamente
if __name__ == '__main__':
    main()