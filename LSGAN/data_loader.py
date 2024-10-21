import os
import torch
from torchvision import datasets, transforms


def get_loader(args):
    train_transforms = []
    if args.dataset != 'mnist':
        train_transforms += [transforms.RandomHorizontalFlip()]
    if args.n_channels == 1:
        train_transforms += [transforms.Grayscale(1)]

    train_transforms += [transforms.Resize([112, 112]),
                         transforms.RandomCrop(112, padding=4, padding_mode='edge'),
                         transforms.ToTensor(),
                         transforms.Normalize([0.5], [0.5])]

    train_transforms = transforms.Compose(train_transforms)

    if args.dataset == 'lsun_church':
        dataset = datasets.LSUN(args.data_path, classes=['church_outdoor_train'], transform=train_transforms)
    elif args.dataset == 'lsun_bedroom':
        dataset = datasets.LSUN(args.data_path, classes=['bedroom_train'], transform=train_transforms)
        samples_to_use = list(range(200000))  # Use only 200k samples
        dataset = torch.utils.data.Subset(dataset, samples_to_use)
    elif args.dataset == 'celeba':
        dataset = datasets.CelebA(args.data_path, split='train', download=True, transform=train_transforms)
    elif args.dataset == 'mnist':
        dataset = datasets.MNIST(args.data_path, train=True, download=True, transform=train_transforms)
    else:
        print("Unknown dataset")
        exit(0)


    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_workers,
                                               drop_last=True)

    return train_loader
