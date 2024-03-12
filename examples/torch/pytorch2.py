import torch
import torch.nn as nn
import torch.optim as optim

EPOCHS_TO_TRAIN = 50000

class XOR(nn.Module):

    def __init__(self):
        super(XOR, self).__init__()
        self.fc1 = nn.Linear(2, 2, True)    
        self.fc2 = nn.Linear(2, 1, True)    

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

xor = XOR()
inputs = torch.Tensor([[0, 0],
    [0, 1],
    [1, 0],
    [1, 1]])


targets = torch.Tensor([[0],[1],[1],[0]])

criterion = nn.MSELoss()
optimizer = optim.SGD(xor.parameters(), lr=0.01)

for idx in range(0, EPOCHS_TO_TRAIN):
    for input_, target in zip(inputs, targets):
        optimizer.zero_grad()   
        output = xor(input_)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    
    if idx % 5000 == 0:
        print("Epoch {: >8} Loss: {}".format(idx, loss.item()))


for input_, target in zip(inputs, targets):
    output = xor(input_)
    print(f'In:{input_.data.numpy()}, T:{target.data.numpy()[0]}')
    print(f'Pred:{output.data.numpy()[0]:.5f}, Err:{abs(output.data.numpy()[0] - target.data.numpy()[0]):.5f}')
    print('===')