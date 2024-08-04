import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_weights


def tconv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True):
    module = []
    module.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=k_size, stride=stride, padding=pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)


def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True):
    module = []
    module.append(nn.Conv2d(c_in, c_out, kernel_size=k_size, stride=stride, padding=pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)


class Generator(nn.Module):
    def __init__(self, z_dim=10, n_channels=3, image_size=28, conv_dim=64):
        super().__init__()
        self.image_size = image_size

        self.fc1    = nn.Linear(z_dim, (self.image_size//4)*(self.image_size//4)*conv_dim*2)
        self.tconv2 = tconv_block(conv_dim * 2, conv_dim)
        self.tconv3 = tconv_block(conv_dim, n_channels, use_bn=False)

        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.reshape([x.shape[0], -1, self.image_size//4, self.image_size//4])
        x = F.relu(self.tconv2(x))
        x = torch.tanh(self.tconv3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, n_channels=3, image_size=28, conv_dim=64):
        super().__init__()
        self.conv1 = conv_block(n_channels, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2)
        self.fc3   = nn.Linear((image_size//4)*(image_size//4)*conv_dim*2, 1)

        self.apply(init_weights)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.reshape([x.shape[0], -1])
        x = self.fc3(x)
        return x.squeeze()
