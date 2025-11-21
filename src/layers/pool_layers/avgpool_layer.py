import torch
import torch.nn as nn

class AvgPoolLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(AvgPoolLayer, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)
