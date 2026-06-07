import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from utils import build_hand_graph
from sklearn.model_selection import train_test_split

# ── Carga y prepara el dataset ────────────────────────────────────
def load_dataset():
    df = pd.read_csv('hand_landmarks.csv', header=None)
    
    dataset = []
    edge_index = build_hand_graph()
    
    for _, row in df.iterrows():
        coords = row[:-1].values.reshape(21, 3).astype(np.float32)
        label = int(row[-1])
        
        x = torch.tensor(coords, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    
    return dataset