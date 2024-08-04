import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_weights


class Generator(nn.Module):
    def __init__(self, z_dim=10, n_channels=3, image_size=28, h_dim=1024):
        super().__init__()
        self.image_size = image_size
        self.n_channels = n_channels

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, image_size * image_size * n_channels)

        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x.reshape([-1, self.n_channels, self.image_size, self.image_size])
        return x


class Discriminator(nn.Module):
    def __init__(self, n_channels=3, image_size=28, h_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(image_size * image_size * n_channels, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)

        self.apply(init_weights)

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()
