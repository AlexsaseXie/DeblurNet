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
    default_cell_ = None

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(self.in_channels + self.hidden_channels, 4 * self.hidden_channels, kernel_size=kernel_size, padding=self.padding)

    def forward(self, x, state):
        # x: [b, c, h, w]
        # take h[i], c[i], x as input,
        # returns h[i+1](namely cell's output), c[i+1]
        h, c = torch.chunk(state, 2, dim=1)
        conv = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(conv, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        next_state = torch.cat([next_h, next_c], dim=1)

        return next_state

    @staticmethod
    def init_state(batch_size, hidden_channels, shape):
        height, width = shape
        return Variable(torch.zeros(batch_size, hidden_channels * 2, height, width)).cuda()

    @classmethod
    def default_cell(cls, in_channels, hidden_channels, kernel_size=3):
        if cls.default_cell_ is None:
            cls.default_cell_ = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        return cls.default_cell_


class CNNLayer(nn.Module):
    def __init__(self, shape, channels=6):
        super(CNNLayer, self).__init__()
        self.height, self.width = shape
        self.in_channels = channels
        self.out_channels = channels

        self.input = InputBlock(self.in_channels, 32)
        self.encoder1 = EncoderBlock(32, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.lstm = ConvLSTMCell.default_cell(128, 128)
        self.decoder1 = DecoderBlock(128, 64)
        self.decoder2 = DecoderBlock(64, 32)
        self.output = OutputBlock(32, 3)

    def forward(self, x, state):
        in1 = self.input(x)
        enc1 = self.encoder1(in1)
        enc2 = self.encoder2(enc1)
        next_state = self.lstm(enc2, state)
        lstm_output = torch.chunk(next_state, 2, dim=1)[0]
        dec1 = self.decoder1(lstm_output)
        dec2 = self.decoder2(dec1 + enc1)
        output = self.output(dec2 + in1)
        return output, next_state


class DeblurNetGenerator(nn.Module):
    def __init__(self, shape, num_levels, batch_size, channels=6):
        super(DeblurNetGenerator, self).__init__()
        self.height, self.width = shape
        self.num_levels = num_levels
        self.channels = channels

        self.min_height = self.height / (2 ** self.num_levels)
        self.min_width = self.width / (2 ** self.num_levels)
        self.scaled_shapes = [(int(round(self.min_height * (2 ** i))),
                               int(round(self.min_width * (2 ** i)))) for i in range(self.num_levels)]

        self.layers = [CNNLayer(scaled_shape, self.channels) for scaled_shape in self.scaled_shapes]
        self.state = ConvLSTMCell.init_state(batch_size, 128, self.scaled_shapes[0])

    @staticmethod
    def scale(tensor, shape, batch=True):
        # tensor: [channels, height, width] in [0, 255]
        # when converting PIL image into tensor
        # results lie in [0, 1]

        tr = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(shape),
            transforms.ToTensor()
        ])

        if not batch:
            return tr(tensor)
        else:
            batches = torch.split(tensor, 1, 0)  # tuple
            scaled_batches = [tr(one_tensor) for one_tensor in batches]
            return torch.cat(scaled_batches, dim=0)

    def forward(self, x):
        # x: [batch, channels, height, width]
        pred = x
        preds = []

        for i in range(self.num_levels):
            scaled_x = self.scale(x, self.scaled_shapes[i])
            with torch.no_grad():
                scaled_last_pred = self.scale(pred, self.scaled_shapes[i])
            scaled_state = self.scale(self.state, self.scaled_shapes[i])

            inputs = torch.cat([scaled_x, scaled_last_pred], dim=1)

            pred, self.state = self.layers[i](inputs, scaled_state)
            preds.append(pred)

        return preds[-1]
