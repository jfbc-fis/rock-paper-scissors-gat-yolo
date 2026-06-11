# Rock Paper Scissors — GAT vs YOLOv8

> Real-time gesture recognition system using two competing AI models: a **Graph Attention Network** that reads hand structure, and **YOLOv8** that reads raw pixels. Both predict simultaneously so you can measure how often they agree.

---

## What This Project Does

A two-player Rock Paper Scissors game where two completely different AI models classify the same hand gesture at the same time:

- **Model 1 — GAT (Graph Attention Network):** reads 21 hand landmarks as a graph. Understands that fingertip nodes are connected to knuckle nodes, which connect to the wrist. Learns the *geometry* of each gesture.
- **Model 2 — YOLOv8:** receives the raw image frame. Detects and localizes the gesture directly from pixels using transfer learning from millions of images.
- **Live comparison:** both models predict every frame. The game records how often they agree — this agreement rate is the core experimental result of the project.

---

## Demo

```
ESPACIO = start round   R = restart   Q = quit + show stats
```

At the end of each session a summary window shows per-gesture accuracy and overall agreement between models.

---

## Results

| Model | Accuracy | Epochs | Data |
|-------|----------|--------|------|
| GAT | **93.3%** validation accuracy | 100 | 600 samples (self-collected) |
| YOLOv8 | **94.7%** mAP50 | 30 | Roboflow public dataset |

Neither model is universally better — GAT wins on interpretability and data efficiency; YOLO wins on robustness to lighting and background changes.

---

## Project Structure

```
rock-paper-scissors-gat-yolo/
│
├── src/
│   ├── utils.py                  # HandLandmarkExtractor, normalize_landmarks, build_hand_graph
│   ├── data_collector_screen.py  # Collects landmarks from screen capture → CSV
│   ├── train_gat_model.py        # Trains the GAT model
│   ├── train_yolo_model.py       # Downloads dataset from Roboflow + trains YOLOv8
│   ├── inference_live.py         # Real-time GAT inference (single model)
│   └── game_dual.py              # GAT vs YOLO live game (main application)
│
├── demos/
│   └── 01_tensor_basics.py       # PyTorch tensor fundamentals
│
├── docs/
│   └── git-reference.html        # Conventional Commits reference
│
├── models/                        # Trained weights — not tracked by Git
│   ├── gesture_gat.pt
│   └── yolo_gestures.pt
│
├── data/                          # Datasets — not tracked by Git
│   └── hand_landmarks.csv
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Tech Stack

```
Python        3.10
torch         2.1.1+cu118      # Deep learning engine (CUDA 11.8)
torch-geometric 2.4.0          # Graph Neural Network layers (GATConv)
mediapipe     0.10.8           # Hand landmark extraction
opencv-python 4.8.1.78         # Frame capture and visualization
ultralytics                    # YOLOv8
mss                            # Screen capture (Teams / Zoom compatible)
numpy         1.26.4           # Pinned — incompatible with numpy ≥ 2
roboflow                       # Dataset download for YOLO training
scikit-learn                   # train_test_split
pandas                         # CSV loading
```

> **Important:** `numpy` must stay at `1.26.4`. Installing `mediapipe`, `ultralytics`, or `roboflow` will try to upgrade it. After each `pip install`, run `pip install "numpy<2"` to restore the pinned version.

---

## Setup

### Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA 11.8 (tested on RTX 3060)
- Windows 11 / PowerShell

### Installation

```powershell
# Clone the repository
git clone https://github.com/jfbc-fis/rock-paper-scissors-gat-yolo.git
cd rock-paper-scissors-gat-yolo

# Create and activate virtual environment
py -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch==2.1.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Pin numpy (required after installing mediapipe / ultralytics)
pip install "numpy<2"
pip install opencv-python==4.8.1.78 --force-reinstall --no-deps
```

### Verify GPU is available

```powershell
py -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Usage

All scripts are run from the `src/` directory.

```powershell
cd src
```

### Step 1 — Collect your own dataset

```powershell
py data_collector_screen.py
```

Show your hand to the screen and press:
- `0` → collect Rock samples
- `1` → collect Paper samples
- `2` → collect Scissors samples
- `q` → quit (saves to `data/hand_landmarks.csv`)

Collect at least 200 samples per gesture (600 total). Vary hand position, distance, and angle for better generalization.

### Step 2 — Train the GAT model

```powershell
py train_gat_model.py
```

Trains for 100 epochs. Saves weights to `models/gesture_gat.pt`. Expected validation accuracy: ~93%.

### Step 3 — Train the YOLO model

```powershell
py train_yolo_model.py
```

Downloads the Rock-Paper-Scissors dataset from Roboflow and trains YOLOv8n with transfer learning for 30 epochs. Saves best weights — move them to `models/yolo_gestures.pt`.

```powershell
Move-Item runs\detect\train\weights\best.pt ..\models\yolo_gestures.pt
```

### Step 4 — Test GAT inference alone

```powershell
py inference_live.py
```

Single-model real-time prediction. Shows gesture name and confidence.

### Step 5 — Run the dual model game

```powershell
py game_dual.py
```

Both models predict simultaneously. Press `SPACE` to start a round countdown, `Q` to quit and see the final comparison summary.

---

## How It Works

### GAT Pipeline

```
Screen frame
    └── MediaPipe → 21 landmarks (x, y, z)
         └── normalize_landmarks() → centered on wrist, scaled to [-1, 1]
              └── build_hand_graph() → edge_index [2, N] (anatomical connections)
                   └── GATConv × 2 → global_mean_pool → Linear × 2
                        └── softmax → predicted gesture + confidence
```

### YOLO Pipeline

```
Screen frame
    └── YOLOv8n (CSPDarknet backbone)
         └── detects bounding box + class in one pass
              └── predicted gesture + confidence + box coordinates
```

### Why Graph Attention?

A standard dense network would receive 63 flat numbers with no structural information. GAT encodes the anatomical connections of the hand — the fingertip of the index finger (node 9) attends most to its neighbor (node 8), less to distant nodes. This relational structure improves classification, especially for gestures that differ mainly in finger configuration.

---

## Known Issues

### numpy version conflicts

Several libraries (`mediapipe`, `ultralytics`, `roboflow`) attempt to upgrade numpy to `2.x`, which breaks `torch 2.1.1`. Solution:

```powershell
pip install "numpy<2" --force-reinstall --no-deps
pip install opencv-python==4.8.1.78 --force-reinstall --no-deps
```

### OpenCV GUI error after YOLO install

Ultralytics installs `opencv-python` 4.10+ which lacks Windows GUI support in some builds. Fix:

```powershell
pip install opencv-python==4.8.1.78 --force-reinstall
```

### torch version conflict with ultralytics

After installing ultralytics, verify torch version:

```powershell
py -c "import torch; print(torch.__version__)"
```

If it changed from `2.1.1+cu118`, restore it:

```powershell
pip install torch==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118 --no-deps
```

---

## Model Architecture

### GestureGAT

```python
GestureGAT(
  gat1: GATConv(3 → 64, heads=4)    # input: (x, y, z) per node → 256 features
  gat2: GATConv(256 → 32, heads=4)  # 256 → 128 features
  fc1:  Linear(128 → 64)
  fc2:  Linear(64 → 3)              # output: logits for Rock / Paper / Scissors
)
```

- Input: graph with 21 nodes, each with 3 features (x, y, z)
- Edges: 22 anatomical connections (fingers + knuckles), bidirectional → 44 edges
- Pooling: `global_mean_pool` collapses 21 nodes into a single graph-level vector
- Dropout: 0.3 during training, 0.0 during inference

### YOLOv8n

- Backbone: CSPDarknet (Ultralytics custom architecture)
- Base weights: COCO pretrained (`yolov8n.pt`)
- Fine-tuned: 30 epochs on Rock-Paper-Scissors SXSW dataset (Roboflow)
- Output: bounding box + class + confidence per detection

---

## Dataset

The GAT dataset (`hand_landmarks.csv`) was self-collected using `data_collector_screen.py`:

| Gesture | Samples |
|---------|---------|
| Rock    | 200     |
| Paper   | 200     |
| Scissors| 200     |
| **Total** | **600** |

Each row contains 63 normalized coordinates (21 landmarks × 3 axes) plus a class label. The file is excluded from version control (see `.gitignore`).

The YOLO dataset was downloaded from [Roboflow](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw) (public, CC BY 4.0).

---

## Development Workflow

This project was developed following professional Git practices:

- **Conventional Commits** — `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`
- **Atomic commits** — one logical change per commit
- **Protected files** — models and datasets in `.gitignore` (too large, not reproducible by others)
- **Pinned dependencies** — `requirements.txt` with exact versions for reproducibility

---

## Academic Context

**Course:** Aprendizaje Profundo (Deep Learning)
**Program:** Maestría en Ciencia de Datos
**Institution:** UAEMex
**Year:** 2026

The central research question: *do a geometry-based model (GAT) and a vision-based model (YOLO) agree when classifying the same hand gesture?* The agreement rate measured during live gameplay is the experimental answer.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Francisco Cruz** — [@jfbc-fis](https://github.com/jfbc-fis)