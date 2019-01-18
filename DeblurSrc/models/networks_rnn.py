import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

import os
from .networks import get_norm_layer, weights_init


def define_RNN_G(opt):
    netG = None
    use_gpu = len(opt.gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=opt.norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if opt.which_model_netG == 'rnn_reuse':
        netG = DeblurNetGeneratorReuse(shape=(opt.fineSize, opt.fineSize), num_levels=3, batch_size=opt.batchSize, gpu=use_gpu)
    else:
        netG = DeblurNetGenerator(shape=(opt.fineSize, opt.fineSize), num_levels=3, batch_size=opt.batchSize, gpu=use_gpu)

    if len(opt.gpu_ids) > 0:
        netG.cuda(opt.gpu_ids[0])
    netG.apply(weights_init)
    return netG


class ResBlock(nn.Module):
    # reflection padding
    # instance norm
    # conv -> norm -> relu -> conv
    def __init__(self, channels, kernel_size=5, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

        self.conv_network = nn.Sequential(
            self.conv1,
            self.norm,
            self.relu,
            self.conv2
        )

    def forward(self, x):
        # x: [batch, channels, height, width]
        # out: [batch, channels, height, width]
        out = x + self.conv_network(x)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, res_kernel_size=5, stride=2):
        super(EncoderBlock, self).__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=self.stride, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.res1 = ResBlock(out_channels, res_kernel_size)
        self.res2 = ResBlock(out_channels, res_kernel_size)
        self.res3 = ResBlock(out_channels, res_kernel_size)

        self.network = nn.Sequential(
            self.conv,
            self.relu,
            self.res1,
            self.res2,
            self.res3
        )

    def forward(self, x):
        # x: [batch, in_channels, height, width]
        # out: [batch, out_channels, height / 2, width / 2]
        out = self.network(x)
        return out


class InputBlock(EncoderBlock):
    def __init__(self, in_channels, out_channels, kernel_size=5, res_kernel_size=5, stride=1):
        super(InputBlock, self).__init__(in_channels, out_channels, kernel_size, res_kernel_size, stride)

    def forward(self, x):
        # x: [batch, in_channels, height, width]
        # out: [batch, out_channels, height, width]
        out = self.network(x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, res_kernel_size=5, stride=2):
        super(DecoderBlock, self).__init__()
        self.stride = stride
        self.padding = (self.stride * (in_channels - 1) - 2 * in_channels + kernel_size) // 2
        self.res1 = ResBlock(in_channels, res_kernel_size)
        self.res2 = ResBlock(in_channels, res_kernel_size)
        self.res3 = ResBlock(in_channels, res_kernel_size)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=self.stride, padding=self.padding)
        self.relu = nn.ReLU()

        self.network = nn.Sequential(
            self.res1,
            self.res2,
            self.res3,
            self.deconv,
            self.relu
        )

    def forward(self, x):
        # x: [batch, in_channels, height, width]
        # out: [batch, out_channels, height * 2, width * 2]
        out = self.network(x)
        return out


class OutputBlock(nn.Module):
    # no activation at last
    def __init__(self, in_channels, out_channels, kernel_size=5, res_kernel_size=5, stride=1):
        super(OutputBlock, self).__init__()
        self.res1 = ResBlock(in_channels, res_kernel_size)
        self.res2 = ResBlock(in_channels, res_kernel_size)
        self.res3 = ResBlock(in_channels, res_kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)

        self.network = nn.Sequential(
            self.res1,
            self.res2,
            self.res3,
            self.conv
        )

    def forward(self, x):
        # x: [batch, in_channels, height, width]
        # out: [batch, out_channels, height, width]
        out = self.network(x)
        return out


class ConvLSTMCell(nn.Module):
    _default_cell = None

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(self.in_channels + self.hidden_channels, 4 * self.hidden_channels, kernel_size=kernel_size, padding=self.padding)

    def forward(self, *inputs):
        x, h, c = inputs
        # x: [batch, in_channels, height, width]
        # state: [batch, hidden_channels * 2, height, width]
        # h, c: [batch, hidden_channels, height, width]

        # take h[i], c[i], x as input,
        # returns h[i+1](namely cell's output), c[i+1]

        conv = self.conv(torch.cat([x, h], dim=1))  # conv: [batch, in_channels + hidden_channels, height, width]
                                                    # ->    [batch, hidden_channels * 4, height, width]
        i, f, o, g = torch.chunk(conv, 4, dim=1)    # i, f, o, g: [batch, hidden_cannels, height, width]
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)

        return next_h, next_c  # next_h, next_c: same with h, c

    @staticmethod
    def init_state(batch_size, hidden_channels, shape, gpu=True):
        height, width = shape
        if gpu:
            return (Variable(torch.zeros(batch_size, hidden_channels, height, width)).cuda(),
                    Variable(torch.zeros(batch_size, hidden_channels, height, width)).cuda())
        else:
            return (Variable(torch.zeros(batch_size, hidden_channels, height, width)),
                    Variable(torch.zeros(batch_size, hidden_channels, height, width)))

    @classmethod
    def default_cell(cls, in_channels, hidden_channels, kernel_size=3):
        if cls._default_cell is None:
            cls._default_cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        return cls._default_cell


class CNNLayer(nn.Module):
    def __init__(self, channels=6):
        super(CNNLayer, self).__init__()
        self.in_channels = channels
        self.out_channels = channels

        self.input = InputBlock(self.in_channels, 32)
        self.encoder1 = EncoderBlock(32, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.lstm = ConvLSTMCell.default_cell(128, 128)
        self.decoder1 = DecoderBlock(128, 64)
        self.decoder2 = DecoderBlock(64, 32)
        self.output = OutputBlock(32, 3)

    def forward(self, *inputs):
        x, h, c = inputs
        # x: [batch, channels, height, width]
        # h, c: [batch, hidden_channels, height, width]
        in1 = self.input(x)  # in1: [batch, 32, height, width]
        enc1 = self.encoder1(in1)  # enc1: [batch, 64, height / 2, width / 2]
        enc2 = self.encoder2(enc1)  # enc2: [batch, 128, height / 4, width / 4]
        next_h, next_c = self.lstm(enc2, h, c)  # next_h, next_c: [batch, 128, height / 4, width / 4]
        dec1 = self.decoder1(next_h)  # dec1: [batch, 64, height / 2, width / 2]
        dec2 = self.decoder2(dec1 + enc1)  # dec2: [batch, 32, height, width]
        out = self.output(dec2 + in1)  # out: [batch, 32, height, width]
        return out, next_h, next_c


class DeblurNetGeneratorReuse(nn.Module):
    def __init__(self, shape, num_levels, batch_size, channels=6, gpu=True):
        super(DeblurNetGeneratorReuse, self).__init__()
        self.height, self.width = shape
        self.num_levels = num_levels
        self.batch_size = batch_size
        self.channels = channels
        self.gpu = gpu

        self.min_height = self.height / (2 ** self.num_levels)
        self.min_width = self.width / (2 ** self.num_levels)
        self.scaled_shapes = [(int(round(self.min_height * (2 ** i))),
                               int(round(self.min_width * (2 ** i)))) for i in range(self.num_levels)]

        self.layer = CNNLayer(self.channels)

    def forward(self, *xs):
        # xs: (x_0, x_1, x_2)
        # x: [batch, 3, height, width]
        pred = xs[0]
        batch_size = pred.size(0)
        preds = []
        state = ConvLSTMCell.init_state(batch_size, 128, self.scaled_shapes[0], self.gpu)  # h, c

        for i in range(self.num_levels):
            x_shape = (xs[i].shape[-2], xs[i].shape[-1])

            scaled_last_pred = F.interpolate(pred.detach().cuda(), size=x_shape, mode='bilinear', align_corners=False)
            scaled_h = F.interpolate(state[0], size=(xs[i].shape[-2] // 4, xs[i].shape[-1] // 4), mode='bilinear', align_corners=False)
            scaled_c = F.interpolate(state[1], size=(xs[i].shape[-2] // 4, xs[i].shape[-1] // 4), mode='bilinear', align_corners=False)
            inputs = torch.cat([xs[i], scaled_last_pred], dim=1)

            pred, next_h, next_c = self.layer(inputs, scaled_h, scaled_c)
            state = next_h, next_c
            preds.append(pred)

        return tuple(preds)


class DeblurNetGenerator(nn.Module):
    def __init__(self, shape, num_levels, batch_size, channels=6, gpu=True):
        super(DeblurNetGenerator, self).__init__()
        self.height, self.width = shape
        self.num_levels = num_levels
        self.channels = channels

        self.min_height = self.height / (2 ** self.num_levels)
        self.min_width = self.width / (2 ** self.num_levels)
        self.scaled_shapes = [(int(round(self.min_height * (2 ** i))),
                               int(round(self.min_width * (2 ** i)))) for i in range(self.num_levels)]

        self.layers = [CNNLayer(self.channels) for scaled_shape in self.scaled_shapes]
        self.state = ConvLSTMCell.init_state(batch_size, 128, self.scaled_shapes[0], gpu)

    def forward(self, *xs):
        # x: [batch, channels, height, width]
        pred = xs[0]
        preds = []

        for i in range(self.num_levels):
            x_shape = (xs[i].shape[-2], xs[i].shape[-1])

            scaled_last_pred = F.interpolate(pred.detach().cuda(), size=x_shape, mode='bilinear', align_corners=False)
            #print(pred.device)
            scaled_h = F.interpolate(self.state[0], size=(xs[i].shape[-2] // 4, xs[i].shape[-1] // 4), mode='bilinear', align_corners=False)
            scaled_c = F.interpolate(self.state[1], size=(xs[i].shape[-2] // 4, xs[i].shape[-1] // 4), mode='bilinear', align_corners=False)
            inputs = torch.cat([xs[i], scaled_last_pred], dim=1)

            pred, next_h, next_c = self.layers[i](inputs, scaled_h, scaled_c)
            self.state = next_h, next_c
            preds.append(pred)

        return tuple(preds)


def test_loss(pred):
    return torch.mean(torch.mean(pred, dim=1))


def test_model(device=None):
    if device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    batch = 2
    net = DeblurNetGeneratorReuse((256, 256), 3, batch)
    net = net.cuda()
    for param in net.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam([param for param in net.parameters() if param.requires_grad], weight_decay=0, lr=1e-4)
    net.train()

    for i in range(3):
        optimizer.zero_grad()
        input0 = torch.randn(batch, 3, 64, 64)
        input1 = torch.randn(batch, 3, 128, 128)
        input2 = torch.randn(batch, 3, 256, 256)
        input0 = Variable(torch.cuda.FloatTensor(input0.cuda()))
        input1 = Variable(torch.cuda.FloatTensor(input1.cuda()))
        input2 = Variable(torch.cuda.FloatTensor(input2.cuda()))
        output = net(input0, input1, input2)

        loss = test_loss(output[0])
        loss.backward()
        optimizer.step()

        #print(output[0].shape)
        print(float(loss))


if __name__ == '__main__':
    test_model(0)
