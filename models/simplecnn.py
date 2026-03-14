import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*self.img_size*self.img_size, self.num_classes)
        self.relu = nn.ReLU()
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x