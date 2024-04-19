import torch.nn as nn
import torch.nn.functional as F


class ImpalaResidualBlock(nn.Module):
    def __init__(self, num_channels, use_bn=False):
        super(ImpalaResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1
        )
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = x
        if self.use_bn:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class ImpalaConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False):
        super(ImpalaConvSequence, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ImpalaResidualBlock(out_channels, use_bn=use_bn)
        self.res2 = ImpalaResidualBlock(out_channels, use_bn=use_bn)

    def forward(self, x):
        out = self.conv(x)
        out = self.maxpool(out)
        out = self.res1(out)
        out = self.res2(out)
        return out
