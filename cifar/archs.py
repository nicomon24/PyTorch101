'''
    Collection of architectures
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class BaselineNet(nn.Module):

    def __init__(self):
        super(BaselineNet, self).__init__()
        # Define the network modules
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Defines what happens in the forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class VggNet(nn.Module):

    def __init__(self):
        super(VggNet, self).__init__()
        # Define the network modules
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.mp3 = nn.MaxPool2d(2, stride=2)
        self.c4 = nn.Conv2d(64, 128, 3, padding=1)
        self.c5 = nn.Conv2d(128, 128, 3, padding=1)
        self.mp6 = nn.MaxPool2d(2, stride=2)
        self.c7 = nn.Conv2d(128, 256, 3, padding=1)
        self.c8 = nn.Conv2d(256, 256, 3, padding=1)
        self.c9 = nn.Conv2d(256, 256, 3, padding=1)
        self.c10 = nn.Conv2d(256, 256, 3, padding=1)
        self.mp11 = nn.MaxPool2d(2, stride=2)
        self.fc12 = nn.Linear(256 * 4 * 4, 256)
        self.fc13 = nn.Linear(256, 10)

    def forward(self, x):
        # Convolutions
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.mp3(x)
        x = F.relu(self.c4(x))
        x = F.relu(self.c5(x))
        x = self.mp6(x)
        x = F.relu(self.c7(x))
        x = F.relu(self.c8(x))
        x = F.relu(self.c9(x))
        x = F.relu(self.c10(x))
        x = self.mp11(x)
        # Fully connected
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
