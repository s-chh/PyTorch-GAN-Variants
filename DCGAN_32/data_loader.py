import os
import torch
from torchvision import datasets, transforms


def get_loader(args):
    train_transforms     = []
    if args.dataset      in ['fashionmnist', 'cifar10']:
        train_transforms += [transforms.RandomHorizontalFlip()]
    if args.n_channels   == 1:
        train_transforms += [transforms.Grayscale(1)]

    train_transforms += [transforms.Resize([32, 32]),
                         transforms.RandomCrop(32, padding=2, padding_mode='edge'),
                         transforms.ToTensor(),
                         transforms.Normalize([0.5], [0.5])]

    train_transforms = transforms.Compose(train_transforms)

    if args.dataset   == 'mnist':
        train = datasets.MNIST(args.data_path, train=True, download=True, transform=train_transforms)
    elif args.dataset == 'fashionmnist':
        train = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=train_transforms)
    elif args.dataset == 'svhn':
        train = datasets.SVHN(args.data_path, split='train', download=True, transform=train_transforms)
    elif args.dataset == 'usps':
        train = datasets.USPS(args.data_path, train=True, download=True, transform=train_transforms)
    elif args.dataset == 'cifar10':
        train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=train_transforms)
    else:
        print("Unknown dataset")
        exit(0)

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_workers,
                                               drop_last=True)

    return train_loader
