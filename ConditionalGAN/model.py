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
    def __init__(self, z_dim=10, n_classes=10, y_embed_dim=5, n_channels=3, conv_dim=64):
        super().__init__()
        self.y_embedding = nn.Embedding(n_classes, y_embed_dim)
        self.tconv1 = tconv_block(z_dim + y_embed_dim, conv_dim * 4, pad=0)
        self.tconv2 = tconv_block(conv_dim * 4, conv_dim * 2)
        self.tconv3 = tconv_block(conv_dim * 2, conv_dim)
        self.tconv4 = tconv_block(conv_dim, n_channels, use_bn=False)

        self.apply(init_weights)

    def forward(self, x, y):
        x = x.reshape([x.shape[0], -1, 1, 1])
        y_embed = self.y_embedding(y)
        y_embed = y_embed.reshape([y_embed.shape[0], -1, 1, 1])
        x = torch.cat((x, y_embed), dim=1)
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = torch.tanh(self.tconv4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, n_classes=10, n_channels=3, conv_dim=64):
        super().__init__()
        self.image_size = 32
        self.y_embedding = nn.Embedding(n_classes, self.image_size * self.image_size)
        self.conv1 = conv_block(n_channels + 1, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4)
        self.conv4 = conv_block(conv_dim * 4, 1, k_size=4, stride=1, pad=0, use_bn=False)

        self.apply(init_weights)

    def forward(self, x, y):
        y_embed = self.y_embedding(y)
        y_embed = y_embed.reshape([y_embed.shape[0], 1, self.image_size, self.image_size])
        x = torch.cat((x, y_embed), dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.conv4(x).squeeze()
        return x
