import torch


a = torch.randint(-1, 2, size=(4, 4))
b = torch.randint(-1, 2, size=(4, 4))
logic = torch.logical_or(a, b)
print(a)
print(b)
print(logic)
