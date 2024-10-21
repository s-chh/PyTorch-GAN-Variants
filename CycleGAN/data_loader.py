import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os


def get_dataset(ds, ds_path, transform):
    ds_full_path = os.path.join(ds_path, ds)
    if ds.lower() == 'svhn':
        return datasets.SVHN(ds_full_path, split='train', download=True, transform=transform)
    elif ds.lower() == 'mnist':
        return datasets.MNIST(ds_full_path, train=True, download=True, transform=transform)
    elif ds.lower() == 'usps':
        return datasets.USPS(ds_full_path, train=True, download=True, transform=transform)
    else:
        return datasets.ImageFolder(ds_full_path, transform=transform)


def get_loaders(ds_path='./data', batch_size=128, image_size=32, a_ds='svhn', b_ds='mnist'):
    mean = np.array([0.5])
    std = np.array([0.5])

    transform = transforms.Compose([transforms.Resize([image_size, image_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    a_ds = get_dataset(a_ds, ds_path, transform=transform)
    b_ds = get_dataset(b_ds, ds_path, transform=transform)

    a_ds_loader = torch.utils.data.DataLoader(dataset=a_ds,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              drop_last=True)

    b_ds_loader = torch.utils.data.DataLoader(dataset=b_ds,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              drop_last=True)
    return a_ds_loader, b_ds_loader
