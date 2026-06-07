# ═══════════════════════════════════════════════════════════════
# game_dual.py
# El archivo estrella del proyecto.
# Dos modelos de IA completamente diferentes compiten en tiempo
# real clasificando el mismo gesto — GAT desde grafos,
# YOLOv8 desde píxeles — y medimos cuánto coinciden.
# ═══════════════════════════════════════════════════════════════

import cv2                          # captura y visualización
import torch                        # ejecutar el modelo GAT
import numpy as np                  # manipulación de arrays
import mss                          # captura de pantalla
import time                         # cuenta regresiva
import mediapipe as mp              # detección de manos
from collections import deque       # suavizado de predicciones
from enum import Enum, auto         # máquina de estados del juego
from ultralytics import YOLO        # modelo YOLOv8
from torch_geometric.data import Data, Batch  # grafo para GAT
from utils import HandLandmarkExtractor, normalize_landmarks, build_hand_graph
from train_gat_model import GestureGAT


# ── CONFIGURACIÓN GLOBAL ──────────────────────────────────────────

# Los gestos del juego — índice numérico → nombre
GESTURES = {0: 'Piedra', 1: 'Papel', 2: 'Tijera'}

# Reglas del juego: qué gesto vence a cuál
# BEATS['Piedra'] == 'Tijera' → piedra aplasta tijera
BEATS = {'Piedra': 'Tijera', 'Papel': 'Piedra', 'Tijera': 'Papel'}

# Colores BGR para cada gesto en pantalla
COLORS = {
    'Piedra': (60, 60, 220),    # rojo
    'Papel':  (60, 180, 60),    # verde
    'Tijera': (220, 160, 0),    # azul
    None:     (180, 180, 180)   # gris → sin detección
}

# Mapeo de nombres YOLO → nuestros nombres
# YOLO fue entrenado con nombres en inglés — los traducimos
YOLO_MAP = {
    'rock': 'Piedra', 'paper': 'Papel', 'scissors': 'Tijera',
    'piedra': 'Piedra', 'papel': 'Papel', 'tijera': 'Tijera',
    'Rock': 'Piedra', 'Paper': 'Papel', 'Scissors': 'Tijera',
}

# Cuenta regresiva del juego — texto y duración en segundos
SEQUENCE = [
    ('PIEDRA',    1.0),
    ('PAPEL',     1.0),
    ('o TIJERA!', 1.0),
    ('¡YA!',      0.6),
]


# ── MÁQUINA DE ESTADOS ────────────────────────────────────────────
# El juego siempre está en exactamente uno de estos estados.
# Es imposible estar en dos al mismo tiempo — evita bugs.
class Phase(Enum):
    WAITING  = auto()   # esperando que el jugador presione ESPACIO
    SEQUENCE = auto()   # mostrando cuenta regresiva PIEDRA/PAPEL/TIJERA
    FREEZE   = auto()   # congelando el gesto en el último frame
    REVEAL   = auto()   # mostrando resultado WIN/LOSE/TIE
    GAMEOVER = auto()   # alguien llegó al puntaje máximo


# ── CAPÍTULO 1: Clasificador GAT ─────────────────────────────────
# Carga el modelo que entrenamos con nuestro propio dataset
# Predice gestos desde grafos de landmarks
class GATClassifier:
    def __init__(self, model_path):
        # Usa GPU si está disponible — RTX 3060 en tu caso
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Crea la arquitectura vacía con dropout=0.0
        # dropout=0.0 porque en inferencia queremos predicciones estables
        # model.eval() ya lo desactiva, pero lo ponemos explícito por seguridad
        self.model = GestureGAT(num_classes=3, dropout=0.0).to(self.device)

        # Carga los pesos entrenados desde el archivo .pt
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Modo evaluación — desactiva dropout definitivamente
        self.model.eval()
        print(f"GAT cargado en {self.device}")

    @torch.no_grad()  # no calcular gradientes — solo predecir
    def predict(self, landmarks):
        # 1. Normaliza: centra en muñeca y escala a [-1, 1]
        normalized = normalize_landmarks(landmarks)

        # 2. Convierte a tensor y mueve a GPU
        x = torch.tensor(normalized, dtype=torch.float).to(self.device)

        # 3. Obtiene las conexiones anatómicas de la mano
        edge_index = build_hand_graph().to(self.device)

        # 4. Empaqueta en objeto Data — la "caja" de PyTorch Geometric
        data  = Data(x=x, edge_index=edge_index)

        # 5. Batch de 1 grafo — el modelo espera batches
        batch = Batch.from_data_list([data])

        # 6. Predice y convierte logits a probabilidades con softmax
        probs = torch.softmax(self.model(batch), dim=1)[0]
        label = int(probs.argmax())

        # 7. Devuelve nombre del gesto y confianza
        return GESTURES[label], float(probs[label])


# ── CAPÍTULO 2: Clasificador YOLO ────────────────────────────────
# Carga el modelo entrenado con transfer learning desde imágenes
# Predice gestos directamente desde píxeles — sin landmarks
class YOLOClassifier:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.names = self.model.names  # diccionario de clases del modelo
        print(f"YOLO cargado: {list(self.names.values())}")

    def predict(self, region_bgr):
        # Pasa la imagen por YOLO con confianza mínima de 35%
        results   = self.model(region_bgr, verbose=False, conf=0.35)
        best_conf  = 0.0
        best_class = None

        # Busca la detección con mayor confianza
        for r in results:
            for box in r.boxes:
                name   = self.names[int(box.cls)].lower()
                conf   = float(box.conf)
                # Traduce nombre YOLO → nuestro nombre
                mapped = YOLO_MAP.get(name)
                if mapped and conf > best_conf:
                    best_conf  = conf
                    best_class = mapped

        return best_class, best_conf


# ── CAPÍTULO 3: Detector de dos manos ────────────────────────────
# Detecta hasta 2 manos y las ordena de izquierda a derecha
# para asignar consistentemente Jugador 1 y Jugador 2
class MultiHandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode        = False,
            max_num_hands            = 2,      # detecta ambos jugadores
            min_detection_confidence = 0.65,
            min_tracking_confidence  = 0.50,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame_rgb):
        """Detecta manos y las ordena izquierda → derecha."""
        results = self.hands.process(frame_rgb)
        hands   = []

        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                # Extrae solo (x, y) — suficiente para ubicar en pantalla
                coords = np.array([[l.x, l.y] for l in lms.landmark])
                # Guarda coordenadas y posición x de la muñeca para ordenar
                hands.append((coords, coords[0, 0]))

        # Ordena por posición x de la muñeca
        # → mano izquierda siempre es Jugador 1
        # → mano derecha siempre es Jugador 2
        hands.sort(key=lambda t: t[1])
        return [h[0] for h in hands]

    def draw(self, frame, landmarks_list):
        """Dibuja esqueletos: verde=Jugador1, azul=Jugador2."""
        colors = [(80, 220, 80), (80, 80, 220)]
        h, w   = frame.shape[:2]

        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20)
        ]

        for i, lms in enumerate(landmarks_list):
            color = colors[i % 2]
            pts   = [(int(x * w), int(y * h)) for x, y in lms]
            for s, e in CONNECTIONS:
                cv2.line(frame, pts[s], pts[e], color, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)


# ── CAPÍTULO 4: Interfaz visual ───────────────────────────────────
# Dibuja todo el HUD sobre el frame capturado
# Es una función independiente — no necesita estado propio
def draw_hud(frame, players, phase, seq_word, stats):
    h, w = frame.shape[:2]
    mid  = w // 2

    # Línea divisoria — separa el área de cada jugador
    cv2.line(frame, (mid, 0), (mid, h), (100, 100, 100), 2)

    for i, p in enumerate(players):
        x1 = 0 if i == 0 else mid
        x2 = mid if i == 0 else w
        cx = x1 + (x2 - x1) // 2

        # ── Encabezado oscuro con nombre y puntaje ──
        cv2.rectangle(frame, (x1, 0), (x2, 55), (30, 30, 30), -1)
        cv2.putText(frame, f"Jugador {i+1}", (x1 + 10, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (235, 235, 235), 2)

        # Puntaje en dorado alineado a la derecha
        score_text = str(p['score'])
        sw = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0][0]
        cv2.putText(frame, score_text, (x2 - sw - 10, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 210, 255), 3)

        gat_g  = p['gat_gesture']
        yolo_g = p['yolo_gesture']
        gat_c  = p['gat_conf']
        yolo_c = p['yolo_conf']

        # ── Badge GAT — verde ──
        # Muestra qué predijo el modelo de grafos
        cv2.rectangle(frame, (x1+8, h-130), (x1+185, h-82), (40,40,40), -1)
        cv2.rectangle(frame, (x1+8, h-130), (x1+185, h-82), (100,220,100), 1)
        cv2.putText(frame, "GAT", (x1+14, h-112),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,220,100), 1)
        if gat_g:
            cv2.putText(frame, gat_g, (x1+14, h-88),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7,
                       COLORS.get(gat_g,(255,255,255)), 2)
            cv2.putText(frame, f"{gat_c:.0%}", (x1+145, h-88),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

        # ── Badge YOLO — azul ──
        # Muestra qué predijo el modelo de píxeles
        cv2.rectangle(frame, (x1+193, h-130), (x1+370, h-82), (40,40,40), -1)
        cv2.rectangle(frame, (x1+193, h-130), (x1+370, h-82), (100,100,255), 1)
        cv2.putText(frame, "YOLO", (x1+199, h-112),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255), 1)
        if yolo_g:
            cv2.putText(frame, yolo_g, (x1+199, h-88),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7,
                       COLORS.get(yolo_g,(255,255,255)), 2)
            cv2.putText(frame, f"{yolo_c:.0%}", (x1+330, h-88),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

        # ── Indicador de acuerdo entre modelos ──
        # Esta es la métrica clave de la tesis:
        # ¿Coinciden GAT y YOLO en su predicción?
        if gat_g and yolo_g:
            agree = gat_g == yolo_g
            ac    = (0, 210, 80) if agree else (0, 80, 220)
            al    = "ACUERDO" if agree else "DIFIEREN"
            asz   = cv2.getTextSize(al, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(frame, al, (cx - asz[0]//2, h-55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ac, 2)

        # ── Resultado de la ronda ──
        # Solo visible en Phase.REVEAL y Phase.GAMEOVER
        if phase in (Phase.REVEAL, Phase.GAMEOVER) and p['result']:
            rc = (0,210,80)  if p['result']=='WIN'  else \
                 (40,40,200) if p['result']=='LOSE' else (180,160,30)
            rsz = cv2.getTextSize(p['result'],
                                  cv2.FONT_HERSHEY_DUPLEX, 2.0, 4)[0]
            cv2.putText(frame, p['result'],
                       (cx - rsz[0]//2, h//2 + 10),
                       cv2.FONT_HERSHEY_DUPLEX, 2.0, rc, 4)

            # Gesto árbitro grande — GAT es el árbitro oficial
            if gat_g:
                gsz = cv2.getTextSize(gat_g,
                                      cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
                cv2.putText(frame, gat_g,
                           (cx - gsz[0]//2, h//2 - 40),
                           cv2.FONT_HERSHEY_DUPLEX, 1.5,
                           COLORS.get(gat_g,(255,255,255)), 3)

    # ── HUD central — cambia según la fase ──
    if phase == Phase.WAITING:
        # Instrucción para el usuario
        msg = "ESPACIO para jugar"
        msz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame, msg, (w//2 - msz[0]//2, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,100), 2)

    elif phase == Phase.SEQUENCE:
        # Cuenta regresiva grande en el centro
        ssz = cv2.getTextSize(seq_word,
                              cv2.FONT_HERSHEY_DUPLEX, 2.5, 5)[0]
        cv2.putText(frame, seq_word,
                   (w//2 - ssz[0]//2, h//2 + 20),
                   cv2.FONT_HERSHEY_DUPLEX, 2.5, (20,210,255), 5)

    elif phase == Phase.GAMEOVER:
        # Instrucciones para nueva partida o salir
        msg = "R=nueva partida  Q=salir"
        msz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.putText(frame, msg, (w//2 - msz[0]//2, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 1)


# ── CAPÍTULO 5: Loop principal ────────────────────────────────────
def main():
    # Rutas relativas desde src/ hacia models/
    GAT_PATH  = '../models/gesture_gat.pt'
    YOLO_PATH = '../models/yolo_gestures.pt'

    print("Cargando modelos...")
    gat_clf  = GATClassifier(GAT_PATH)
    yolo_clf = YOLOClassifier(YOLO_PATH)
    detector = MultiHandDetector()
    print("Modelos cargados correctamente")

    # Estado inicial de cada jugador
    # Usamos diccionario simple — no necesita métodos propios
    def new_player():
        return {
            'score': 0, 'result': '',
            'gat_gesture': None, 'gat_conf': 0.0,
            'yolo_gesture': None, 'yolo_conf': 0.0,
            # deque(maxlen=9) → guarda las últimas 9 predicciones
            # para suavizar y evitar parpadeos por frames ruidosos
            'gat_history':  deque(maxlen=9),
            'yolo_history': deque(maxlen=9),
        }

    players = [new_player(), new_player()]

    # Estado del juego
    phase     = Phase.WAITING
    seq_index = 0
    seq_end   = 0.0
    cur_word  = ""
    WIN_SCORE = 3  # primero en llegar a 3 puntos gana
    stats     = {'rounds': 0, 'agreements': 0}

    print("Iniciando captura de pantalla...")
    print("ESPACIO=jugar | R=reiniciar | Q=salir")

    with mss.MSS() as sct:
        # Captura el monitor principal completo
        monitor = sct.monitors[1]

        while True:
            # ── 1. Captura y convierte el frame ──
            screenshot = sct.grab(monitor)
            frame      = np.array(screenshot)
            frame      = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now        = time.time()

            # ── 2. Detecta manos y dibuja esqueletos ──
            hands = detector.detect(frame_rgb)
            detector.draw(frame, hands)

            # ── 3. Clasifica gestos con ambos modelos ──
            # Solo clasifica en fases activas — no en REVEAL o GAMEOVER
            active = phase in (Phase.WAITING, Phase.SEQUENCE, Phase.FREEZE)

            for i, p in enumerate(players):
                if i < len(hands) and active:
                    lms = hands[i]

                    # GAT necesita (x, y, z) — MediaPipe dio (x, y)
                    # Agregamos columna de ceros como coordenada z
                    lms_3d   = np.column_stack([lms, np.zeros(len(lms))])
                    g_g, g_c = gat_clf.predict(lms_3d)

                    # Agrega a historial y toma el más frecuente
                    # Esto suaviza las predicciones — evita parpadeos
                    p['gat_history'].append(g_g)
                    p['gat_gesture'] = max(set(p['gat_history']),
                                          key=list(p['gat_history']).count)
                    p['gat_conf'] = g_c

                    # YOLO predice en su mitad de la pantalla
                    h_f, w_f = frame.shape[:2]
                    mid      = w_f // 2
                    region   = frame[:, 0:mid] if i==0 else frame[:, mid:w_f]
                    y_g, y_c = yolo_clf.predict(region)

                    if y_g:
                        p['yolo_history'].append(y_g)
                        p['yolo_gesture'] = max(set(p['yolo_history']),
                                               key=list(p['yolo_history']).count)
                        p['yolo_conf'] = y_c

            # ── 4. Máquina de estados ──
            if phase == Phase.SEQUENCE:
                # Avanza la cuenta regresiva cuando termina el tiempo
                if now >= seq_end:
                    seq_index += 1
                    if seq_index < len(SEQUENCE):
                        cur_word, dur = SEQUENCE[seq_index]
                        seq_end = now + dur
                    else:
                        # Terminó la cuenta — congela el gesto
                        phase   = Phase.FREEZE
                        seq_end = now + 0.5

            elif phase == Phase.FREEZE:
                # Espera 0.5 segundos y evalúa el ganador
                if now >= seq_end:
                    g = [p['gat_gesture'] for p in players]
                    stats['rounds'] += 1

                    # Aplica las reglas del juego usando BEATS
                    if g[0] == g[1]:
                        players[0]['result'] = 'TIE'
                        players[1]['result'] = 'TIE'
                    elif g[0] and g[1] and BEATS.get(g[0]) == g[1]:
                        players[0]['result'] = 'WIN'
                        players[0]['score'] += 1
                        players[1]['result'] = 'LOSE'
                    elif g[0] and g[1]:
                        players[1]['result'] = 'WIN'
                        players[1]['score'] += 1
                        players[0]['result'] = 'LOSE'

                    # Registra si GAT y YOLO coincidieron
                    for p in players:
                        if p['gat_gesture'] == p['yolo_gesture']:
                            stats['agreements'] += 1
                            break

                    phase = Phase.REVEAL
                    # ¿Alguien llegó al puntaje máximo?
                    if any(p['score'] >= WIN_SCORE for p in players):
                        phase = Phase.GAMEOVER

            # ── 5. Dibuja la interfaz ──
            draw_hud(frame, players, phase, cur_word, stats)

            cv2.imshow('GAT vs YOLO — Piedra Papel Tijera', frame)

            # ── 6. Manejo de teclas ──
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' ') and phase == Phase.WAITING:
                # Inicia la cuenta regresiva
                phase     = Phase.SEQUENCE
                seq_index = 0
                cur_word, dur = SEQUENCE[0]
                seq_end   = now + dur
                for p in players:
                    p['result'] = ''
                    p['gat_history'].clear()
                    p['yolo_history'].clear()

            elif key == ord('r'):
                # Reinicia todo el juego desde cero
                for p in players:
                    p['score']  = 0
                    p['result'] = ''
                    p['gat_history'].clear()
                    p['yolo_history'].clear()
                stats = {'rounds': 0, 'agreements': 0}
                phase = Phase.WAITING

            elif phase == Phase.REVEAL:
                # Cualquier tecla avanza de REVEAL a WAITING
                phase = Phase.WAITING

        cv2.destroyAllWindows()

        # ── 7. Resumen final — la comparativa de tu tesis ──
        if stats['rounds'] > 0:
            acuerdo = stats['agreements'] / stats['rounds'] * 100
            print(f"\n{'='*40}")
            print(f"RESUMEN FINAL — GAT vs YOLO")
            print(f"{'='*40}")
            print(f"Rondas jugadas:    {stats['rounds']}")
            print(f"Acuerdo GAT/YOLO:  {acuerdo:.1f}%")
            print(f"{'='*40}")


# Solo ejecuta si se corre directamente
if __name__ == '__main__':
    main()