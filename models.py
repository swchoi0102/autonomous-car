from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class BaselineNet(nn.Module):

    def __init__(self):
        super(BaselineNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv1_bn = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv2_bn = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv3_bn = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv5_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(1152, 100)
        self.fc1_bn = nn.BatchNorm2d(100)
        self.fc2 = nn.Linear(100, 50)
        self.fc2_bn = nn.BatchNorm2d(50)
        self.fc3 = nn.Linear(50, 10)
        self.fc3_bn = nn.BatchNorm2d(10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.elu(self.conv1_bn(self.conv1(x)))
        x = F.elu(self.conv2_bn(self.conv2(x)))
        x = F.elu(self.conv3_bn(self.conv3(x)))
        x = F.elu(self.conv4_bn(self.conv4(x)))
        x = F.elu(self.conv5_bn(self.conv5(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.elu(self.fc1_bn(self.fc1(x)))
        x = F.elu(self.fc2_bn(self.fc2(x)))
        x = F.elu(self.fc3_bn(self.fc3(x)))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 16, 2),
            conv_dw(16, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            nn.AvgPool2d((2, 4)),
        )

        self.fc1 = nn.Linear(512, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




