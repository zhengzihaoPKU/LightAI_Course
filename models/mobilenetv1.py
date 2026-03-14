import torch
import torch.nn as nn
import torch.nn.functional as F 
# ***********************************first implemetation*****************************
class MobileNetV1_for_imagenet(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNetV1, self).__init__()
        self.conv1 = self._conv_st(3, 32, 2)
        self.conv_dw1 = self._conv_dw(32, 64, 1)
        self.conv_dw2 = self._conv_dw(64, 128, 2)
        self.conv_dw3 = self._conv_dw(128, 128, 1)
        self.conv_dw4 = self._conv_dw(128, 256, 2)
        self.conv_dw5 = self._conv_dw(256, 256, 1)
        self.conv_dw6 = self._conv_dw(256, 512, 2)
        self.conv_dw_x5 = self._conv_x5(512, 512, 5)
        self.conv_dw7 = self._conv_dw(512, 1024, 2)
        self.conv_dw8 = self._conv_dw(1024, 1024, 1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_dw3(x)
        x = self.conv_dw4(x)
        x = self.conv_dw5(x)
        x = self.conv_dw6(x)
        x = self.conv_dw_x5(x)
        x = self.conv_dw7(x)
        x = self.conv_dw8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = torch.softmax(x)
        return x, y

    # dw means depthwise conv
    # Depthwise Conv 这个计算，它的核心在与逐点卷积，在nn.Conv2d的参数中有groups这个参数
    # 默认是groups=1, 意思是分组计算，等于一是就是普通的卷积
    # 当时设置为groups = input_channels，就是深度可分离卷积的depthwise conv
    def _conv_dw(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    # st means standard
    def _conv_st(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    # 中间有重复使用的层设置一个函数循环即可
    def _conv_x5(self, in_channel, out_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(self._conv_dw(in_channel, out_channel, 1))
        return nn.Sequential(*layers)

# ***********************************second implemetation*****************************
class MobileNetV1_for_cifar(nn.Module):
    def __init__(self, input_channel=3, num_classes=100):
        super(MobileNetV1, self).__init__()
        self.num_classes = num_classes
        self.entry = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True))

        self.stage1 = nn.Sequential(
            DSCconv(32, 64, 1),
            DSCconv(64, 128, 2),
            DSCconv(128, 128, 1),
            DSCconv(128, 256, 2),
            DSCconv(256, 256, 1))

        self.stage2 = nn.Sequential(
            DSCconv(256, 512, 2),
            DSCconv(512, 512, 1),
            DSCconv(512, 512, 1),
            DSCconv(512, 512, 1),
            DSCconv(512, 512, 1),
            DSCconv(512, 512, 1))

        self.stage3 = nn.Sequential(
            DSCconv(512, 1024, 2),
            DSCconv(1024, 1024, 1))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # torch.Size([batch, 1024, 1, 1])

        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.entry(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        out = self.softmax(x)
        return out

#  深度可分离卷积 DSC, 深度卷积 Depthwise + 逐点卷积 Pointwise
class DSCconv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(DSCconv, self).__init__()
        self.depthConv = nn.Sequential(  # 深度卷积, (DW+BN+ReLU)
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                      padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True))
        self.pointConv = nn.Sequential(  # 逐点卷积, (PW+BN+ReLU)
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        return x

if __name__ == '__main__':
    net = MobileNetV1_for_cifar(100)
    x = torch.rand(1,3,32,32)
    for name,layer in net.named_children():
        if name != "fc":
            x = layer(x)
            print(name, 'output shape:', x.shape)
        else:
            x = x.view(x.size(0), -1)
            x = layer(x)
            print(name, 'output shape:', x.shape)
