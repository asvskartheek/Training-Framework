import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        if self.transform:
            pass
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pass
