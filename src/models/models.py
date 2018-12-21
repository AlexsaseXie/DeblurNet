import torch.nn as nn


class ResBlock(nn.Module):
    # reflection padding
    # instance norm
    # conv -> norm -> relu -> conv
    def __init__(self, in_channels, out_channels, kernel_size: int, stride=1):
        super(ResBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(kernel_size // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

        self.conv_network = nn.Sequential((
            self.pad1,
            self.conv1,
            self.norm,
            self.relu,
            self.pad2,
            self.conv2
        ))

    def forward(self, x):
        out = x + self.conv_network(x)
        return out


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int, stride=1):
        super(InputBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(kernel_size // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.res1 = ResBlock(out_channels, out_channels, kernel_size, stride)
        self.res2 = ResBlock(out_channels, out_channels, kernel_size, stride)
        self.res3 = ResBlock(out_channels, out_channels, kernel_size, stride)

        self.network = nn.Sequential((
            self.pad1,
            self.conv1,
            self.res1,
            self.res2,
            self.res3
        ))

    def forward(self, x):
        out = self.network(x)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int, stride=2):
        super(EncoderBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(kernel_size // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.res1 = ResBlock(out_channels, out_channels, kernel_size, stride)
        self.res2 = ResBlock(out_channels, out_channels, kernel_size, stride)
        self.res3 = ResBlock(out_channels, out_channels, kernel_size, stride)

        self.network = nn.Sequential((
            self.pad1,
            self.conv1,
            self.res1,
            self.res2,
            self.res3
        ))

    def forward(self, x):
        out = self.network(x)
        return out
