import torch
import torch.nn as nn
import torch.nn.functional as F

class FC_complex(nn.Module):
    def __init__(self, in_channels, img_size, num_classes):
        super(FC_complex, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.fc1=nn.Linear(self.in_channels*self.img_size*self.img_size,1200)
        self.fc2=nn.Linear(1200,1200)
        self.fc3=nn.Linear(1200,num_classes)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.5)
    def forward(self,x):
        x=x.view(-1, self.in_channels*self.img_size*self.img_size)
        x=self.fc1(x)
        x=self.dropout(x)
        x=self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x