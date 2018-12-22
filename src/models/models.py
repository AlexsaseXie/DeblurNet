import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms


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
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(self.in_channels + self.hidden_channels, 4 * self.hidden_channels, kernel_size=kernel_size, padding=self.padding)

    def forward(self, inputs, state):
        # inputs: [b, c, h, w]
        # take h[i], c[i], x as input,
        # returns h[i+1](namely cell's output), c[i+1]
        h, c = state
        conv = self.conv(torch.cat([inputs, h], dim=1))
        i, f, o, g = torch.chunk(conv, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        next_state = (next_h, next_c)

        return next_state

    @staticmethod
    def init_state(batch_size, hidden_channels, shape):
        height, width = shape
        return (Variable(torch.zeros(batch_size, hidden_channels, height, width)).cuda(),
                Variable(torch.zeros(batch_size, hidden_channels, height, width)).cuda())


class DeblurNetGenerator(nn.Module):
    def __init__(self, shape, num_levels, channels=3):
        super(DeblurNetGenerator, self).__init__()
        self.height, self.width = shape
        self.num_levels = num_levels
        self.in_channels = channels

    @staticmethod
    def scale(tensor, shape):
        # tensor: [channels, height, width] in [0, 255]
        # when converting PIL image into tensor
        # results lie in [0, 1]
        tr = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(shape),
            transforms.ToTensor()
        ])
        return tr(tensor) * 255

    def forward(self, x):
        # x: [batch, channels, height, width]
        scaled_h, scaled_w = self.height / (2 ** self.num_levels), self.width / (2 ** self.num_levels)
        pred = x

        for i in range(self.num_levels):
            scaled_h = int(round(scaled_h))
            scaled_w = int(round(scaled_w))

            scaled_x = self.scale(x, (scaled_h, scaled_w))
            scaled_last_pred = self.scale(pred, (scaled_h, scaled_w))

            inputs = torch.cat([scaled_x, scaled_last_pred], dim=1)
            # ...
            pred = None




