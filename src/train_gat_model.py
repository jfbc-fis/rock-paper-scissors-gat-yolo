# ═══════════════════════════════════════════════════════════════
# train_gat_model.py
# Entrena una Graph Attention Network para clasificar gestos
# de mano en: Piedra (0), Papel (1), Tijera (2)
# ═══════════════════════════════════════════════════════════════

import torch                          # motor de deep learning
import torch.nn.functional as F       # funciones como relu y cross_entropy
import pandas as pd                   # leer el CSV con los landmarks
import numpy as np                    # manipular arrays numéricos
from torch_geometric.nn import GATConv, global_mean_pool
# GATConv      → la capa de atención sobre grafos
# global_mean_pool → colapsa 21 nodos en 1 solo vector por gesto
from torch_geometric.loader import DataLoader
# DataLoader   → agrupa grafos en batches para entrenar eficientemente
from torch_geometric.data import Data
# Data         → la "caja" que empaqueta un grafo completo (nodos + aristas + etiqueta)
from sklearn.model_selection import train_test_split
# train_test_split → divide el dataset en 80% entrenamiento / 20% validación
from utils import build_hand_graph
# build_hand_graph → genera las conexiones anatómicas de la mano


# ── CAPÍTULO 1: Cargar el dataset ────────────────────────────────
# Lee el CSV y convierte cada fila en un grafo de PyTorch Geometric
def load_dataset():
    # Lee las 600 filas del CSV sin encabezado
    # Cada fila: 63 coordenadas + 1 etiqueta = 64 columnas
    df = pd.read_csv('hand_landmarks.csv', header=None)

    # El grafo de conexiones es siempre el mismo — la anatomía no cambia
    # Se calcula UNA sola vez y se reutiliza para las 600 muestras
    edge_index = build_hand_graph()

    dataset = []

    for _, row in df.iterrows():
        # row.values[:-1] → toma los 63 números (excluye la etiqueta)
        # .reshape(21, 3)  → los reorganiza en matriz [21 nodos × 3 coordenadas]
        # .astype(float32) → PyTorch necesita float32, no float64
        coords = row.values[:-1].reshape(21, 3).astype(np.float32)

        # La última columna es la etiqueta: 0=piedra, 1=papel, 2=tijera
        label = int(row.values[-1])

        # Convierte las coordenadas a tensor de PyTorch
        # x tiene shape [21, 3] → un vector de 3 features por nodo
        x = torch.tensor(coords, dtype=torch.float)

        # La etiqueta también va en tensor — dtype=long porque es un índice
        y = torch.tensor([label], dtype=torch.long)

        # Data es la "caja" que empaqueta todo el grafo:
        # x          → features de los 21 nodos
        # edge_index → las conexiones anatómicas de la mano
        # y          → la etiqueta correcta (piedra/papel/tijera)
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)

    # Devuelve lista de 600 objetos Data — uno por muestra
    return dataset


# ── CAPÍTULO 2: La arquitectura del modelo ───────────────────────
# GestureGAT es una red neuronal que entiende grafos
# Hereda de torch.nn.Module para obtener todos los superpoderes de PyTorch
class GestureGAT(torch.nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        # Activa los superpoderes de torch.nn.Module
        super().__init__()

        # CAPA 1 de atención sobre grafos
        # Entrada: 3 features por nodo (x, y, z)
        # Salida:  64 features × 4 cabezas de atención = 256 features
        # Cada cabeza aprende a prestar atención a diferentes relaciones
        self.gat1 = GATConv(3, 64, heads=4, dropout=dropout)

        # CAPA 2 de atención sobre grafos
        # Entrada: 256 features (salida de gat1)
        # Salida:  32 features × 4 cabezas = 128 features
        self.gat2 = GATConv(256, 32, heads=4, dropout=dropout)

        # CAPA DENSA 1: reduce de 128 a 64 features
        self.fc1 = torch.nn.Linear(128, 64)

        # CAPA DENSA 2: de 64 features a 3 clases (piedra/papel/tijera)
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        # Desempaqueta el grafo
        # x     → features de los nodos [num_nodos, 3]
        # ei    → conexiones entre nodos [2, num_aristas]
        # batch → indica a qué grafo pertenece cada nodo
        #         (necesario cuando hay múltiples grafos en un batch)
        x, ei, batch = data.x, data.edge_index, data.batch

        # Pasa por la primera capa GAT + activación ReLU
        # ReLU elimina valores negativos — introduce no-linealidad
        x = F.relu(self.gat1(x, ei))

        # Pasa por la segunda capa GAT + activación ReLU
        x = F.relu(self.gat2(x, ei))

        # global_mean_pool: colapsa los 21 nodos en 1 solo vector
        # promediando sus features — convierte el grafo en un vector
        # Shape: [21, 128] → [1, 128] por cada gesto del batch
        x = global_mean_pool(x, batch)

        # Capa densa con ReLU
        x = F.relu(self.fc1(x))

        # Capa final — devuelve 3 números (logits) sin activación
        # El número más alto indica el gesto predicho
        return self.fc2(x)


# ── CAPÍTULO 3: Una epoch de entrenamiento ───────────────────────
def train(model, loader, optimizer):
    # model.train() activa el dropout — introduce ruido para evitar overfitting
    model.train()
    total_loss = 0

    for data in loader:
        # Mueve el batch a la GPU (RTX 3060)
        data = data.cuda()

        # Limpia los gradientes del batch anterior
        # Sin esto los gradientes se acumularían y corromperían el aprendizaje
        optimizer.zero_grad()

        # Paso hacia adelante: el modelo predice los gestos
        out = model(data)

        # Calcula el error entre predicción y etiqueta real
        # cross_entropy es la función de pérdida estándar para clasificación
        loss = F.cross_entropy(out, data.y.squeeze())

        # Paso hacia atrás: PyTorch calcula cómo ajustar cada peso
        loss.backward()

        # El optimizador aplica los ajustes a los pesos
        optimizer.step()

        total_loss += loss.item()

    # Devuelve el loss promedio de todos los batches
    return total_loss / len(loader)


# ── CAPÍTULO 4: Evaluación ────────────────────────────────────────
def evaluate(model, loader):
    # model.eval() desactiva el dropout — predicciones deterministas
    model.eval()
    correct = 0
    total   = 0

    # torch.no_grad() le dice a PyTorch que NO calcule gradientes
    # Solo estamos prediciendo, no entrenando — ahorra memoria y tiempo
    with torch.no_grad():
        for data in loader:
            data = data.cuda()
            out  = model(data)

            # argmax → toma el índice del valor más alto
            # Si out = [0.1, 0.8, 0.1] → pred = 1 (papel)
            pred = out.argmax(dim=1)

            # Compara predicción con etiqueta real
            correct += (pred == data.y.squeeze()).sum().item()
            total   += data.y.size(0)

    # Devuelve accuracy: fracción de predicciones correctas
    return correct / total


# ── CAPÍTULO 5: El programa principal ────────────────────────────
if __name__ == '__main__':
    print("Cargando dataset...")
    dataset = load_dataset()

    # Divide: 80% para entrenar, 20% para validar
    # random_state=42 garantiza que la división sea siempre la misma
    train_data, val_data = train_test_split(
        dataset, test_size=0.2, random_state=42
    )

    # DataLoader agrupa los grafos en batches de 16
    # shuffle=True → mezcla los datos en cada epoch para mejor aprendizaje
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=16, shuffle=False)

    # Crea el modelo y lo mueve a la GPU
    model     = GestureGAT(num_classes=3).cuda()

    # Adam es el optimizador más usado en Deep Learning
    # lr=0.001 → tamaño del paso de aprendizaje (learning rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Entrenando con {len(train_data)} muestras, validando con {len(val_data)}")
    print("-" * 50)

    # Entrena por 100 epochs — 100 pasadas completas por el dataset
    for epoch in range(1, 101):
        loss = train(model, train_loader, optimizer)
        acc  = evaluate(model, val_loader)

        # Imprime progreso cada 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

    # Guarda solo los pesos del modelo — no la arquitectura completa
    # state_dict() es un diccionario con todos los pesos entrenados
    torch.save(model.state_dict(), 'gesture_gat.pt')
    print("\nModelo guardado como gesture_gat.pt")