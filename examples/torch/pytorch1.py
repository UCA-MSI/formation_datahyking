import torch
torch.manual_seed(42)

device = torch.device('cuda:0')
A = torch.randn(4, 4)
A.to(device)
Ainv = torch.linalg.inv(A)
r = torch.dist(A @ Ainv, torch.eye(4))
print(r)