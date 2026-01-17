import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(784, 10)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x