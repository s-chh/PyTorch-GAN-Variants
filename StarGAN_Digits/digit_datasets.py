import os
import cv2
import PIL
import torch
import tarfile
import skimage
import numpy as np
import urllib.request
import torch.nn as nn
from torchvision import datasets, transforms


class MNIST(nn.Module):
    def __init__(self, data_path, label=-100):
        super().__init__()

        self.dataset = datasets.MNIST(data_path, train=True, download=True)
        self.label   = label

        self.in_transform  = transforms.Compose([transforms.Resize([32, 32]), transforms.Grayscale(3)])
        self.out_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, _ = self.dataset.__getitem__(index)

        img = self.in_transform(img)
        img = self.out_transform(img)

        return img, self.label

class NegMNIST(nn.Module):
    def __init__(self, data_path, label=-100):
        super().__init__()

        self.dataset = datasets.MNIST(data_path, train=True, download=True)
        self.label   = label

        self.in_transform  = transforms.Compose([transforms.Resize([32, 32]), transforms.Grayscale(3)])
        self.out_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, _ = self.dataset.__getitem__(index)

        img = self.in_transform(img)
        img = PIL.ImageOps.invert(img)
        img = self.out_transform(img)

        return img, self.label

class EdgeMNIST(nn.Module):
    def __init__(self, data_path, label=-100):
        super().__init__()

        self.dataset = datasets.MNIST(data_path, train=True, download=True)
        self.label   = label

        self.in_transform = transforms.Compose([transforms.Resize([32, 32]), transforms.Grayscale(3)])
        self.out_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, _ = self.dataset.__getitem__(index)
        
        img = self.in_transform(img)
        img = np.array(img)

        dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
        img = dilation - img
        img = PIL.Image.fromarray(img)
        img = self.out_transform(img)

        return img, self.label


class ColorMNIST(nn.Module):
    def __init__(self, data_path, label=-100):
        super().__init__()

        self.dataset = datasets.MNIST(data_path, train=True, download=True)
        self.label   = label

        self.in_transform   = transforms.Compose([transforms.Resize([32, 32]), transforms.Grayscale(3)])
        self.out_transform1 = transforms.ToTensor()
        self.out_transform2 = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, _ = self.dataset.__getitem__(index)
        
        img = self.in_transform(img)

        img = self.out_transform1(img)
        img = torch.rand(3, 1, 1) * img + torch.rand(3, 1, 1) * (1-img)
        torch.clamp_(img, 0, 1)
        img = self.out_transform2(img)

        return img, self.label


class MNISTM(nn.Module):
    def __init__(self, data_path, label=-100):
        super().__init__()

        self.dataset = datasets.MNIST(data_path, train=True, download=True)
        self.label   = label

        self.in_transform  = transforms.Compose([transforms.Resize([32, 32]), transforms.Grayscale(3)])
        self.out_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        background_path = os.path.join(data_path, "BSR_bsds500.tgz")
        if not os.path.isfile(background_path):
            print("Downloading background images for MNIST-M...")
            background_url = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'
            urllib.request.urlretrieve(background_url, background_path)

        print("Processing background images for MNIST-M...")
        f = tarfile.open(background_path)
        background_files = []
        for name in f.getnames():
            if name.startswith('BSR/BSDS500/data/images/train/'):
                background_files.append(name)

        self.background_data = []
        for name in background_files:
            try:
                fp = f.extractfile(name)
                bg_img = skimage.io.imread(fp)
                self.background_data.append(bg_img)
            except:
                continue

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, _ = self.dataset.__getitem__(index)

        img = self.in_transform(img)

        background_image = np.random.randint(low=0, high=len(self.background_data), size=(1))[0]
        background_image = self.background_data[background_image]

        img = np.array(img)
        img = (img > 0).astype(float) * 255

        x_start = np.random.randint(0, background_image.shape[0]-img.shape[0])
        y_start = np.random.randint(0, background_image.shape[1]-img.shape[1])
        background_patch = background_image[x_start:x_start+img.shape[0], y_start:y_start+img.shape[1]]
        img = np.abs(background_patch-img).astype(np.uint8)

        img = self.out_transform(img)

        return img, self.label


class USPS(nn.Module):
    def __init__(self, data_path, label=-100):
        super().__init__()

        self.dataset = datasets.USPS(data_path, train=True, download=True)
        self.label   = label

        self.in_transform  = transforms.Compose([transforms.Resize([32, 32]), transforms.Grayscale(3)])
        self.out_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, _ = self.dataset.__getitem__(index)
        
        img = self.in_transform(img)
        img = self.out_transform(img)

        return img, self.label


class SVHN(nn.Module):
    def __init__(self, data_path, label=-100):
        super().__init__()

        self.dataset = datasets.SVHN(data_path, split='train', download=True)
        self.label = label

        self.in_transform  = transforms.Resize([32, 32])
        self.out_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img, _ = self.dataset.__getitem__(index)
        
        img = self.in_transform(img)
        img = self.out_transform(img)

        return img, self.label


class OtherDigits(nn.Module):
    def __init__(self, data_path, label=-100):
        super().__init__()

        self.dataset = datasets.ImageFolder(data_path)
        self.label = label

        self.in_transform  = transforms.Resize([32, 32])
        self.out_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    
    
    def __getitem__(self, index):
        img, _ = self.dataset.__getitem__(index)
        
        img = self.in_transform(img)
        img = self.out_transform(img)

        return img, self.label

