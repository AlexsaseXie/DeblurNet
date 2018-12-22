import torch
import torch.nn as nn
from torch.autograd import Variable


class ResBlock(nn.Module):
    # reflection padding
    # instance norm
    # conv -> norm -> relu -> conv
    def __init__(self, in_channels, out_channels, kernel_size: int, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

        self.conv_network = nn.Sequential((
            self.conv1,
            self.norm,
            self.relu,
            self.conv2
        ))

    def forward(self, x):
        out = x + self.conv_network(x)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 5, stride=2):
        super(EncoderBlock, self).__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=self.stride, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.res1 = ResBlock(out_channels, out_channels, kernel_size, self.stride)
        self.res2 = ResBlock(out_channels, out_channels, kernel_size, self.stride)
        self.res3 = ResBlock(out_channels, out_channels, kernel_size, self.stride)

        self.network = nn.Sequential((
            self.conv,
            self.relu,
            self.res1,
            self.res2,
            self.res3
        ))

    def forward(self, x):
        out = self.network(x)
        return out


class InputBlock(EncoderBlock):
    def __init__(self, in_channels, out_channels, kernel_size: int = 5, stride=1):
        super(InputBlock, self).__init__(in_channels, out_channels, kernel_size, stride)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 4, stride=2):
        super(DecoderBlock, self).__init__()
        self.stride = stride
        self.padding = (self.stride * (in_channels - 1) - 2 * in_channels + kernel_size) // 2
        self.res1 = ResBlock(in_channels, out_channels, kernel_size, stride)
        self.res2 = ResBlock(out_channels, out_channels, kernel_size, stride)
        self.res3 = ResBlock(out_channels, out_channels, kernel_size, stride)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=self.stride, padding=self.padding)
        self.relu = nn.ReLU()

        self.network = nn.Sequential((
            self.res1,
            self.res2,
            self.res3,
            self.deconv,
            self.relu
        ))

    def forward(self, x):
        out = self.network(x)
        return out


class OutputBlock(nn.Module):
    # no activation at last
    def __init__(self, in_channels, out_channels, kernel_size: int = 5, stride=1):
        super(OutputBlock, self).__init__()
        self.res1 = ResBlock(in_channels, out_channels, kernel_size, stride)
        self.res2 = ResBlock(out_channels, out_channels, kernel_size, stride)
        self.res3 = ResBlock(out_channels, out_channels, kernel_size, stride)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)

        self.network = nn.Sequential((
            self.res1,
            self.res2,
            self.res3,
            self.conv
        ))

    def forward(self, x):
        out = self.network(x)
        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, in_shape: tuple, in_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = in_shape
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(self.in_channels + self.hidden_channels, 4 * self.hidden_channels, kernel_size=kernel_size, padding=self.padding)

    def forward(self, inputs, state):
        # inputs: [b, c, h, w]
        h, c = state
        conv = self.conv(torch.cat([inputs, h], dim=1))
        i, f, o, g = torch.chunk(conv, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)

        return next_h, next_c

    @staticmethod
    def init_state(batch_size, hidden_channels, shape):
        height, width = shape
        return Variable(torch.zeros(batch_size, hidden_channels, height, width)).cuda()
