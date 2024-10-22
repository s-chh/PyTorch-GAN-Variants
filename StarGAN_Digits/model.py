import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
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


class Generator(nn.Module):
    def __init__(self, in_channels=3, n_domains=5, image_size=32, out_channels=3, conv_dim=64):
        super().__init__()
        self.image_size = image_size
        self.embed_layer = nn.Embedding(n_domains, image_size**2)

        self.conv1  = conv_block(in_channels+1, conv_dim, k_size=5, stride=1, pad=2, use_bn=True)
        self.conv2  = conv_block(conv_dim, conv_dim * 2, k_size=4, stride=2, pad=1, use_bn=True)
        self.conv3  = conv_block(conv_dim * 2, conv_dim * 4, k_size=4, stride=2, pad=1, use_bn=True)
        self.res4   = ResBlock(conv_dim * 4)
        self.res5   = ResBlock(conv_dim * 4)
        self.res6   = ResBlock(conv_dim * 4)
        self.tconv7 = conv_block(conv_dim * 4, conv_dim * 2, k_size=4, stride=2, pad=1, use_bn=True, transpose=True)
        self.tconv8 = conv_block(conv_dim * 2, conv_dim, k_size=4, stride=2, pad=1, use_bn=True, transpose=True)
        self.conv9  = conv_block(conv_dim, out_channels, k_size=5, stride=1, pad=2, use_bn=False)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, target_dm=None):
        if target_dm is None:
            target_dm = torch.ones(x.shape[0])
        target_dm = target_dm.long()
        embed = self.embed_layer(target_dm).reshape([-1, 1, self.image_size, self.image_size])
        x = torch.cat((x, embed), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.res4(x))
        x = F.relu(self.res5(x))
        x = F.relu(self.res6(x))
        x = F.relu(self.tconv7(x))
        x = F.relu(self.tconv8(x))
        x = torch.tanh(self.conv9(x))
        return x


class Critic(nn.Module):
    def __init__(self, channels=3, n_domains=5, image_size=32, conv_dim=64):
        super().__init__()
        self.conv1 = conv_block(channels, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2, use_bn=False)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4, use_bn=False)
        self.conv4 = conv_block(conv_dim * 4, conv_dim * 8, use_bn=False)

        self.gan = conv_block(conv_dim * 8, 1, k_size=3, stride=1, pad=1, use_bn=False)
        self.cls = conv_block(conv_dim * 8, n_domains, k_size=image_size//16, stride=1, pad=0, use_bn=False)

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
        x = F.leaky_relu(self.conv4(x), alpha)
        gan_out = self.gan(x)
        cls_out = self.cls(x)
        
        return gan_out, cls_out.squeeze()

