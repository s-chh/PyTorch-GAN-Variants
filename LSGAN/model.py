import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_weights


def tconv_block(c_in, c_out, k_size=3, stride=2, pad=1, out_pad=1, use_bn=True):
    module = []
    module.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=k_size, stride=stride, padding=pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)


def conv_block(c_in, c_out, k_size=5, stride=2, pad=2, use_bn=True):
    module = []
    module.append(nn.Conv2d(c_in, c_out, kernel_size=k_size, stride=stride, padding=pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)


def linear_block(fc_in, fc_out, use_bn=True):
    module = []
    module.append(nn.Linear(fc_in, fc_out, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm1d(fc_out))
    return nn.Sequential(*module)


class Generator(nn.Module):
    def __init__(self, z_dim=10, n_channels=3, conv_dim=64):
        super().__init__()
        self.fc1 = linear_block(z_dim, 7*7*256)
        self.tconv2 = tconv_block(256, 256, stride=2, pad=1, out_pad=1)
        self.tconv3 = tconv_block(256, 256, stride=1, pad=1)
        self.tconv4 = tconv_block(256, 256, stride=2, pad=1, out_pad=1)
        self.tconv5 = tconv_block(256, 256, stride=1, pad=1)
        self.tconv6 = tconv_block(256, 128, stride=2, pad=1, out_pad=1)
        self.tconv7 = tconv_block(128, 64, stride=2, pad=1, out_pad=1)
        self.tconv8 = tconv_block(64, n_channels, stride=1, pad=1, use_bn=False)

        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.reshape([x.shape[0], 256, 7, 7])
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))
        x = F.relu(self.tconv5(x))
        x = F.relu(self.tconv6(x))
        x = F.relu(self.tconv7(x))
        x = torch.tanh(self.tconv8(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, n_channels=3, conv_dim=64):
        super().__init__()
        self.conv1 = conv_block(n_channels, 64, use_bn=False)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.fc5   = linear_block(7*7*512, 1, use_bn=False)

        self.apply(init_weights)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.reshape([x.shape[0], -1])
        x = self.fc5(x)
        return x.squeeze()

