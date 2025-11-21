import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_identity=True):
        super().__init__()
        self.use_identity = use_identity and in_channels == out_channels and stride == 1

        # 3x3 conv branch
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3x3 = nn.BatchNorm2d(out_channels)

        # 1x1 conv branch (shortcut if dims mismatch)
        if not self.use_identity:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn1x1 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1x1 = None
            self.bn1x1 = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn3x3(self.conv3x3(x))

        if self.conv1x1 is not None:
            out += self.bn1x1(self.conv1x1(x))
        elif self.use_identity:
            out += x

        return self.relu(out)
