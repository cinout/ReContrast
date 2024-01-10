import torch
import torch.nn.functional as F

a = torch.randint(0, 6, size=(3, 2, 2), dtype=torch.float)
b = torch.randint(0, 6, size=(3, 2, 2), dtype=torch.float)
print(a)
print(b)

a = torch.mean(a, dim=(1, 2))
b = torch.mean(b, dim=(1, 2))
print(a)
print(b)

sim = F.cosine_similarity(a, b, dim=0)
print(sim)
