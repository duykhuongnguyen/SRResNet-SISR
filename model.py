### Module imports ###
import math

import torch
from torch import nn
from torchsummary import summary


### Global Variables ###


### Class declarations ###
class SRResNet(nn.Module):

    def __init__(self, scale_factor, num_residual=5):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(SRResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        residual = [ResidualBlock(64) for _ in range(num_residual)]
        self.residual = nn.Sequential(*residual)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        upsample = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        upsample.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.upsample = nn.Sequential(*upsample)

    def forward(self, x):
        conv1 = self.conv1(x)
        residual = self.residual(conv1)
        conv2 = self.conv2(residual)
        upsample = self.upsample(conv1 + conv2)
        return (torch.tanh(upsample) + 1) / 2


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class UpsampleBLock(nn.Module):

    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


### Function declarations ###
if __name__ == '__main__':
    model = SRResNet(4, 5).cuda()
    summary(model, (3, 22, 22))
