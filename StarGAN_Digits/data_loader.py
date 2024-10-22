import torch
from digit_datasets import *

def get_loader(digit_datasets, data_path='./data', batch_size=128, n_workers=4):

    combined_dataset = []
    for i, dataset in enumerate(digit_datasets):
        if dataset == 'mnist':
            combined_dataset.append(MNIST(data_path, label=i))
        elif dataset == 'nmnist':
            combined_dataset.append(NegMNIST(data_path, label=i))
        elif dataset == 'emnist':
            combined_dataset.append(EdgeMNIST(data_path, label=i))
        elif dataset == 'cmnist':
            combined_dataset.append(ColorMNIST(data_path, label=i))
        elif dataset == 'mnistm':
            combined_dataset.append(MNISTM(data_path, label=i))
        elif dataset == 'usps':
            combined_dataset.append(USPS(data_path, label=i))
        elif dataset == 'svhn':
            combined_dataset.append(SVHN(data_path, label=i))
        else:
            combined_dataset.append(OtherDataset(os.path.join(data_path, dataset), label=i))

    datsets = torch.utils.data.ConcatDataset(combined_dataset)

    ds_loader = torch.utils.data.DataLoader(dataset=datsets,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=n_workers,
                                              drop_last=True)
    return ds_loader
