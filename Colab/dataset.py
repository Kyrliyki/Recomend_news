"""
PyTorch Dataset для MIND данных
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class MindDataset(Dataset):
    """PyTorch Dataset для MIND данных"""

    def __init__(self, df):
        if len(df) == 0:
            self.data = {
                'userIdx': torch.tensor([]),
                'click': torch.tensor([])
            }
        else:
            self.data = {
                'userIdx': torch.tensor(df.userIdx.values.astype(np.int64)),
                'click': torch.tensor(df.click.values.astype(np.int64))
            }

    def __len__(self):
        return len(self.data['userIdx'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}