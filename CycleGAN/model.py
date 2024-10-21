import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, output_padding=pad, bias=not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = conv_block(channels, channels, k_size=3, stride=1, pad=1, use_bn=True)
        self.conv2 = conv_block(channels, channels, k_size=3, stride=1, pad=1, use_bn=True)

    def __call__(self, x):
        x = F.relu(self.conv1(x))
        return x + self.conv2(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv_block(channels, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4)
        self.conv4 = conv_block(conv_dim * 4, 1, k_size=3, stride=1, pad=1, use_bn=False)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        alpha = 0.2
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = F.leaky_relu(self.conv3(x), alpha)
        x = self.conv4(x)
        x = x.reshape([x.shape[0], -1]).mean(1)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, conv_dim=64):
        super(Generator, self).__init__()
        self.conv1 = conv_block(in_channels, conv_dim, k_size=5, stride=1, pad=2, use_bn=True)
        self.conv2 = conv_block(conv_dim, conv_dim * 2, k_size=3, stride=2, pad=1, use_bn=True)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4, k_size=3, stride=2, pad=1, use_bn=True)
        self.res4 = ResBlock(conv_dim * 4)
        self.tconv5 = conv_block(conv_dim * 4, conv_dim * 2, k_size=3, stride=2, pad=1, use_bn=True, transpose=True)
        self.tconv6 = conv_block(conv_dim * 2, conv_dim, k_size=3, stride=2, pad=1, use_bn=True, transpose=True)
        self.conv7 = conv_block(conv_dim, out_channels, k_size=5, stride=1, pad=2, use_bn=False)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.res4(x))
        x = F.relu(self.tconv5(x))
        x = F.relu(self.tconv6(x))
        x = torch.tanh(self.conv7(x))
        return x
