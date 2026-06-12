import numpy as np
import torch
from torch.utils.data import Dataset

class BadmintonDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])