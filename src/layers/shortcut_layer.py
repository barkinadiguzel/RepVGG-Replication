import torch
import torch.nn as nn

class ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        return self.shortcut(x)
