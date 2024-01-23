import torch
import torch.nn.functional as F
import os
from dataset import MyDummyDataset
from recontrast_mvtecloco import InfiniteDataloader
from torch.utils.data import Dataset, TensorDataset

aa = [
    {"name": "puta", "image": torch.tensor([9.0])},
    {"name": "uiop", "image": torch.tensor([2.0])},
]

aa = MyDummyDataset(aa)

train_dataloader = torch.utils.data.DataLoader(
    aa,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

for a in train_dataloader:
    print(a)
