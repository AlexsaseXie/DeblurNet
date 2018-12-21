import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, use_bias, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(kernel_size // 2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=0, bias=use_bias)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=0, bias=use_bias)
        self.conv_layer = nn.Sequential([
            self.pad1,
            self.conv1,
            self.relu,
            self.conv2
        ])

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out
