import torch
import torch.nn as nn

from src.layers.conv_layer import ConvLayer
from src.layers.residual_block import ResidualBlock
from src.layers.pool_layers.maxpool_layer import MaxPoolLayer
from src.layers.pool_layers.avgpool_layer import AvgPoolLayer
from src.layers.flatten_layer import FlattenLayer
from src.layers.fc_layer import FCLayer
from src.config import REP_VGG_CONFIG, NUM_CLASSES, USE_IDENTITY

class RepVGG(nn.Module):
    def __init__(self, version="A", num_classes=NUM_CLASSES):
        super().__init__()
        cfg = REP_VGG_CONFIG[version]
        num_blocks = cfg["num_blocks"]
        width_mult = cfg["width_mult"]

        self.stage0 = ConvLayer(
            in_channels=3, 
            out_channels=int(64 * width_mult[0]), 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            use_bn=True,
            activation=True
        )

        self.stages = nn.ModuleList()
        in_channels = int(64 * width_mult[0])
        stage_out_channels = [64, 128, 256, 512, 512]

        for i, num_layer in enumerate(num_blocks):
            layers = []
            out_channels = int(stage_out_channels[i] * width_mult[i])
            for j in range(num_layer):
                stride = 2 if j == 0 and i != 0 else 1
                layers.append(ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_identity=USE_IDENTITY
                ))
                in_channels = out_channels
            self.stages.append(nn.Sequential(*layers))

        # Head
        self.global_pool = AvgPoolLayer(kernel_size=7, stride=1)
        self.flatten = FlattenLayer()
        self.fc = FCLayer(in_features=in_channels, out_features=num_classes)

    def forward(self, x):
        x = self.stage0(x)
        for stage in self.stages:
            x = stage(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
