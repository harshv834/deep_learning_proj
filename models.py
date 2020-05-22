
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dlc_practical_prologue as prologue
#%matplotlib inline
N=1000
from torch.utils.data import DataLoader, Dataset
import tqdm
from torch.autograd import Variable



class CompareNet1(nn.Module):
    def __init__(self):
        super(CompareNet1, self).__init__()
        self.base = nn.Sequential(nn.Linear(196, 64), nn.LeakyReLU(), nn.Linear(64, 32))
        self.comparator = nn.Linear(64, 2)
        self.classifier = nn.Linear(32, 10)  # for auxillary loss removal, remove this

    def forward(self, x):
        x_0 = x[:, 0, :, :].flatten(1)
        x_1 = x[:, 1, :, :].flatten(1)
        x_0 = F.leaky_relu(self.base(x_0))
        x_1 = F.leaky_relu(self.base(x_1))
        sign = F.leaky_relu(self.comparator(torch.cat([x_0, x_1], dim=1)))
        digit_0 = F.leaky_relu(self.classifier(x_0))
        digit_1 = F.leaky_relu(self.classifier(x_1))
        return sign, digit_0, digit_1


class CompareNet11(nn.Module):  # without weight sharing
    def __init__(self):
        super(CompareNet11, self).__init__()
        self.base = nn.Sequential(nn.Linear(392, 196), nn.LeakyReLU(), nn.Linear(196, 64), nn.LeakyReLU())
        self.comparator = nn.Linear(64, 2)
        self.classifier = nn.Linear(64, 100)

    def forward(self, x):
        x = x.view(-1, 392)
        x = F.leaky_relu(self.base(x))
        sign = F.leaky_relu(self.comparator(x))
        num = F.leaky_relu(self.classifier(x))
        return sign, num


class CompareNet12(nn.Module):  # without auxillary loss
    def __init__(self):
        super(CompareNet12, self).__init__()
        self.base = nn.Sequential(nn.Linear(196, 64), nn.LeakyReLU(), nn.Linear(64, 32))
        self.comparator = nn.Linear(64, 2)

    def forward(self, x):
        x_0 = x[:, 0, :, :].flatten(1)
        x_1 = x[:, 1, :, :].flatten(1)
        x_0 = F.leaky_relu(self.base(x_0))
        x_1 = F.leaky_relu(self.base(x_1))
        sign = F.leaky_relu(self.comparator(torch.cat([x_0, x_1], dim=1)))
        return sign


class CompareNet2(nn.Module):
    def __init__(self, batch_normalization=False):
        super(CompareNet2, self).__init__()
        if batch_normalization:
            self.base = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3), nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(32),
                                      nn.Conv2d(32, 8, kernel_size=3), nn.MaxPool2d(kernel_size=2))
        else:
            self.base = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3), nn.MaxPool2d(kernel_size=2),
                                      nn.Conv2d(32, 8, kernel_size=3), nn.MaxPool2d(kernel_size=2))
        # self.base = nn.Sequential(nn.Conv2d(1, 8, kernel_size=5),nn.MaxPool2d(kernel_size=5))

        self.comparator = nn.Linear(64, 2)
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        # x = x.flatten(1) # to remove weight sharing
        # x = F.relu(self.base(x))
        x_0 = x[:, :1, :, :]  # for conv2d
        x_1 = x[:, 1:2, :, :]  # for conv2d
        x_0 = F.leaky_relu(self.base(x_0))
        x_1 = F.leaky_relu(self.base(x_1))
        x_0 = x_0.view(-1, 32)
        x_1 = x_1.view(-1, 32)
        sign = F.leaky_relu(self.comparator(torch.cat([x_0, x_1], dim=1)))
        digit_0 = F.leaky_relu(self.classifier(x_0))
        digit_1 = F.leaky_relu(self.classifier(x_1))
        return sign, digit_0, digit_1


class CompareNet21(nn.Module):  # without weight sharing
    def __init__(self, batch_normalization=False):
        super(CompareNet21, self).__init__()
        if batch_normalization:
            self.base = nn.Sequential(nn.Conv2d(2, 32, kernel_size=3), nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(32),
                                      nn.Conv2d(32, 16, kernel_size=3), nn.MaxPool2d(kernel_size=2))
        else:
            self.base = nn.Sequential(nn.Conv2d(2, 32, kernel_size=3), nn.MaxPool2d(kernel_size=2),
                                      nn.Conv2d(32, 16, kernel_size=3), nn.MaxPool2d(kernel_size=2))
        self.comparator = nn.Linear(64, 2)
        self.classifier = nn.Linear(64, 100)

    def forward(self, x):
        x = F.relu(self.base(x))
        x = x.view(-1, 64)
        sign = F.leaky_relu(self.comparator(x))
        num = F.leaky_relu(self.classifier(x))
        return sign, num


class CompareNet22(nn.Module):  # without auxillary loss
    def __init__(self, batch_normalization=False):
        super(CompareNet22, self).__init__()
        if batch_normalization:
            self.base = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3), nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(32),
                                      nn.Conv2d(32, 8, kernel_size=3), nn.MaxPool2d(kernel_size=2))
        else:
            self.base = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3), nn.MaxPool2d(kernel_size=2),
                                      nn.Conv2d(32, 8, kernel_size=3), nn.MaxPool2d(kernel_size=2))

        self.comparator = nn.Linear(64, 2)

    def forward(self, x):
        x_0 = x[:, :1, :, :]
        x_1 = x[:, 1:2, :, :]
        x_0 = F.leaky_relu(self.base(x_0))
        x_1 = F.leaky_relu(self.base(x_1))
        x_0 = x_0.view(-1, 32)
        x_1 = x_1.view(-1, 32)
        sign = F.leaky_relu(self.comparator(torch.cat([x_0, x_1], dim=1)))
        return sign
