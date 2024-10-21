import torch.nn as nn


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)

    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
