import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torchvision import models

non_linearity = partial(F.relu, inplace=True)


class DBlock(nn.Module):
    def __init__(self, channel):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel / 2, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        self.dilate6 = nn.Conv2d(channel, channel, kernel_size=3, dilation=32, padding=32)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = non_linearity(self.dilate1(x))
        dilate2_out = non_linearity(self.dilate2(dilate1_out))
        dilate3_out = non_linearity(self.dilate3(dilate2_out))
        dilate4_out = non_linearity(self.dilate4(dilate3_out))
        dilate5_out = non_linearity(self.dilate5(dilate4_out))
        dilate6_out = non_linearity(self.dilate6(dilate5_out))
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out + dilate6_out
        return out


class DUnet(nn.Module):
    def __init__(self):
        super(DUnet, self).__init__()

        vgg13 = models.vgg13(pretrained=True)

        self.conv1 = vgg13.features[0]
        self.conv2 = vgg13.features[2]
        self.conv3 = vgg13.features[5]
        self.conv4 = vgg13.features[7]
        self.conv5 = vgg13.features[10]
        self.conv6 = vgg13.features[12]

        self.dilate_center = DBlock(512)

        self.up3 = self.conv_stage(512, 256)
        self.up2 = self.conv_stage(256, 128)
        self.up1 = self.conv_stage(128, 64)

        self.trans3 = self.up_sample(512, 256)
        self.trans2 = self.up_sample(256, 128)
        self.trans1 = self.up_sample(128, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True)
        )

    def up_sample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        stage1 = non_linearity(self.conv2(non_linearity(self.conv1(x))))
        stage2 = non_linearity(self.conv4(non_linearity(self.conv3(self.max_pool(stage1)))))
        stage3 = non_linearity(self.conv6(non_linearity(self.conv5(self.max_pool(stage2)))))

        out = self.dilate_center(self.max_pool(stage3))

        out = self.up3(torch.cat((self.trans3(out), stage3), 1))
        out = self.up2(torch.cat((self.trans2(out), stage2), 1))
        out = self.up1(torch.cat((self.trans1(out), stage1), 1))

        out = self.conv_last(out)

        return out
