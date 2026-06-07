# ═══════════════════════════════════════════════════════════════
# train_yolo_model.py
# Descarga dataset de Roboflow y entrena YOLOv8 con transfer
# learning para detectar piedra, papel y tijera desde píxeles.
# ═══════════════════════════════════════════════════════════════

import os
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    # ── CAPÍTULO 1: Descargar el dataset ─────────────────────
    # Roboflow tiene miles de datasets públicos de visión por computadora
    # Este dataset tiene ~2000 imágenes de manos etiquetadas
    print("Descargando dataset de Piedra, Papel o Tijera...")
    rf = Roboflow(api_key="n8LfDsU5EDdNirQPdHwy")
    project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
    version = project.version(1)
    dataset = version.download("yolov8")

    # Roboflow genera automáticamente el archivo data.yaml
    # que describe las clases y rutas del dataset
    ruta_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"Dataset listo en: {dataset.location}")

    # ── CAPÍTULO 2: Transfer Learning ────────────────────────
    # yolov8n.pt = modelo base entrenado con COCO (80 clases)
    # Ya sabe detectar objetos en general — le enseñamos gestos
    # Transfer learning: reutiliza lo aprendido, solo ajusta
    print("Cargando modelo YOLOv8 base...")
    modelo = YOLO("yolov8n.pt")

    # ── CAPÍTULO 3: Entrenamiento ─────────────────────────────
    print("Iniciando entrenamiento con RTX 3060...")
    modelo.train(
        data=ruta_yaml,  # dataset de gestos
        epochs=30,       # 30 pasadas completas
        imgsz=640,       # tamaño estándar de YOLO
        batch=16,        # 16 imágenes por batch
        device=0,        # GPU (RTX 3060)
        workers=2        # hilos del CPU para cargar imágenes
    )
    print("Entrenamiento completado!")

if __name__ == "__main__":
    main()