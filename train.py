import torch

device = torch.device("cuda")
x = torch.tensor([1,2,3,4,5])
x = x.to(device)
print(x)