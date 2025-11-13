import math, os
import torch
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from src.utils.io_helpers import ensure_dir

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y0, y1, z):
        self.x, self.y0, self.y1, self.z = x, y0, y1, z
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i):
        return self.x[i], self.y0[i], self.y1[i], self.z[i]

def make_loader(x, y0, y1, z, batch_size, shuffle=True):
    ds = PairDataset(x, y0, y1, z)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

class BaseTrainer:
    def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator

