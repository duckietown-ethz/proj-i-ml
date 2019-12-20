# -*- coding: utf-8 -*-
#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class CNN(nn.Module):
    def __init__(self, as_gray=True, use_convcoord=True):
        super(OurCNN, self).__init__()

        # Handle dimensions
        if as_gray:
            self.input_channels = 1
        else:
            self.input_channels = 3

        if use_convcoord:
            self.input_channels += 2

        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64))

        self.drop_out_lin1 = nn.Dropout(0.4)
        self.lin1 = nn.Linear(1152 ,512)
        self.drop_out_lin2 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(512, 256)
        self.drop_out_lin3 = nn.Dropout(0.1)
        self.lin3 = nn.Linear(256, 2)


    def forward(self, x):
        # print('dim of input')
        # print(x.size())
        # print(x[0][2][0])
        out = self.layer1(x)
        # print('dim after L1')
        # print(out.size())
        out = self.layer2(out)
        # print('dim after L2')
        # print(out.size())
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        # print('dim after reshape')
        # print(out.size())
        out = self.drop_out_lin1(out)
        out = F.relu(self.lin1(out))
        out = self.drop_out_lin2(out)
        out = F.relu(self.lin2(out))
        out = self.drop_out_lin3(out)
        out = F.tanh(self.lin3(out))

        return out
