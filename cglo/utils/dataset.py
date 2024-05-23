import torch

from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, db):
        super().__init__()
        self.db = db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        return index, self.db[index].to(torch.float)

