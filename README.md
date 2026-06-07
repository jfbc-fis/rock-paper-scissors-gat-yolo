# Rock Paper Scissors — GAT vs YOLOv8

Real-time gesture recognition system using two competing AI models.

## Models
- **GAT (Graph Attention Network):** reads 21 hand landmarks as a graph
- **YOLOv8:** detects gestures directly from raw pixels

## Project Structure
src/                  # source code
├── utils.py          # landmark extraction, normalization, graph builder
├── data_collector_screen.py  # dataset collection tool
├── train_gat_model.py        # GAT training
├── inference_live.py         # real-time GAT inference
└── game_dual.py              # GAT vs YOLO live game
demos/                # learning experiments
docs/                 # documentation and references
models/               # trained model weights (not tracked)
data/                 # datasets (not tracked)
## Tech Stack
- Python 3.10 · PyTorch 2.1.1 · torch-geometric 2.4.0
- MediaPipe 0.10.8 · OpenCV 4.8.1 · Ultralytics YOLOv8
- MSS (screen capture)

## Results
- GAT validation accuracy: **93.3%** (100 epochs)
- Dataset: 600 samples (200 per gesture)

## Status
