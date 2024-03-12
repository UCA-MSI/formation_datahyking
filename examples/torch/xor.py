import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

EPOCHS_TO_TRAIN = 50000

# For clarity purposes, you can define your architecture in
# a class, extending nn.Module.
class XOR(nn.Module):

    def __init__(self):
        # Here you define all the layers, their shape and their type (e.g., Linear)
        super(XOR, self).__init__()
        self.fc1 = nn.Linear(2, 2, True)    # first neuron
        self.fc2 = nn.Linear(2, 1, True)    # second neuron

    def forward(self, x):
        # Back-propagation is offered for free with the backward method.
        # We need to specify the Feed-Forward
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

print("Training loop:")
for idx in range(0, EPOCHS_TO_TRAIN):
    for input_, target in zip(inputs, targets):
        optimizer.zero_grad()   # zero the gradient buffers
        output = xor(input_)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update
    if idx % 5000 == 0:
        print("Epoch {: >8} Loss: {}".format(idx, loss.item()))


for input_, target in zip(inputs, targets):
    output = xor(input_)
    print(f'I:{input_.data.numpy()}, T:{target.data.numpy()[0]}')
    print(f'O:{output.data.numpy()[0]:.5f}, E:{abs(output.data.numpy()[0] - target.data.numpy()[0]):.5f}')
    print('===')