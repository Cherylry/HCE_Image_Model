import os

from torch import nn
import torch
import timm


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            BasicBlock(out_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = timm.create_model('resnet50', pretrained=True, features_only=True)
        self.decoder = nn.Sequential(
            UpBlock(2048, 1024),
            UpBlock(1024, 512),
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, 64),
            nn.Conv2d(64, 3, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    model = MyModel()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
