import torch
import torch.nn.functional as F
import os
from recontrast_mvtecloco import InfiniteDataloader

cool = "datasets/loco/breakfast_box/test/logical_anomalies/073.png"
na = os.path.basename(cool).split(".png")[0]
print(na)
