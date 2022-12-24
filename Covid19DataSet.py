import numpy as np
import torch
from torch.utils.data import Dataset


class Covid19DataSet(Dataset):

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.Tensor(np.array(y))
        self.x = torch.Tensor(np.array(x))

    def __getitem__(self, item):
        if self.y is None:
            return self.x[item]
        else:
            return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)