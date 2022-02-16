"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from functools import partial
from models.moco2_module import MocoV2
from torchvision import models

non_linearity = partial(F.relu, inplace=True)


class DBlockMoreDilate(nn.Module):
    def __init__(self, channel):
        super(DBlockMoreDilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
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
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class DBlock(nn.Module):
    def __init__(self, channel):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = non_linearity(self.dilate1(x))
        dilate2_out = non_linearity(self.dilate2(dilate1_out))
        dilate3_out = non_linearity(self.dilate3(dilate2_out))
        dilate4_out = non_linearity(self.dilate4(dilate3_out))
        # dilate5_out = non_linearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = non_linearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = non_linearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = non_linearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DLinkNet18(nn.Module):
    def __init__(self, backbone_type='imagenet', num_classes=1, num_channels=3):
        super(DLinkNet18, self).__init__()

        filters = [64, 128, 256, 512]

        if backbone_type == 'random':
            resnet = models.resnet18(pretrained=False)
        elif backbone_type == 'imagenet':
            resnet = models.resnet18(pretrained=True)
        elif backbone_type == 'pretrain':
            home_dir = os.environ['HOME']
            ckpt_dir = os.path.join(home_dir, 'checkpoints')
            # ckpt_path = f'{ckpt_dir}/seasonal-contrast/seco_resnet18_100k.ckpt'
            ckpt_path = f'{ckpt_dir}/seasonal-contrast/seco_resnet18_1m.ckpt'
            model = MocoV2.load_from_checkpoint(ckpt_path)
            resnet = deepcopy(model.encoder_q)
            del model
        else:
            raise ValueError()

        if backbone_type == 'pretrain':
            self.first_conv = resnet[0]
            self.first_bn = resnet[1]
            self.first_relu = resnet[2]
            self.first_max_pool = resnet[3]
            self.encoder1 = resnet[4]
            self.encoder2 = resnet[5]
            self.encoder3 = resnet[6]
            self.encoder4 = resnet[7]
        else:
            self.first_conv = resnet.conv1
            self.first_bn = resnet.bn1
            self.first_relu = resnet.relu
            self.first_max_pool = resnet.maxpool
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4

        self.d_block = DBlock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final_deconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = non_linearity
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = non_linearity
        self.final_conv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_max_pool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.d_block(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return torch.sigmoid(out)


class DLinkNet34LessPool(nn.Module):
    def __init__(self, num_classes=1):
        super(DLinkNet34LessPool, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.first_conv = resnet.conv1
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.first_max_pool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.d_block = DBlockMoreDilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final_deconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = non_linearity
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = non_linearity
        self.final_conv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_max_pool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.d_block(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return torch.sigmoid(out)


class DLinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DLinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.first_conv = resnet.conv1
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.first_max_pool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.d_block = DBlock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final_deconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = non_linearity
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = non_linearity
        self.final_conv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_max_pool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.d_block(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return torch.sigmoid(out)


class DLinkNet50(nn.Module):
    def __init__(self, backbone_type='imagenet', num_classes=1):
        super(DLinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]

        if backbone_type == 'random':
            resnet = models.resnet50(pretrained=False)
        elif backbone_type == 'imagenet':
            resnet = models.resnet50(pretrained=True)
        elif backbone_type == 'pretrain':
            home_dir = os.environ['HOME']
            ckpt_dir = os.path.join(home_dir, 'checkpoints')
            # ckpt_path = f'{ckpt_dir}/seasonal-contrast/seco_resnet50_100k.ckpt'
            ckpt_path = f'{ckpt_dir}/seasonal-contrast/seco_resnet50_1m.ckpt'
            model = MocoV2.load_from_checkpoint(ckpt_path)
            resnet = deepcopy(model.encoder_q)
            del model
        else:
            raise ValueError()

        if backbone_type == 'pretrain':
            self.first_conv = resnet[0]
            self.first_bn = resnet[1]
            self.first_relu = resnet[2]
            self.first_max_pool = resnet[3]
            self.encoder1 = resnet[4]
            self.encoder2 = resnet[5]
            self.encoder3 = resnet[6]
            self.encoder4 = resnet[7]
        else:
            self.first_conv = resnet.conv1
            self.first_bn = resnet.bn1
            self.first_relu = resnet.relu
            self.first_max_pool = resnet.maxpool
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4

        self.d_block = DBlockMoreDilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final_deconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = non_linearity
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = non_linearity
        self.final_conv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_max_pool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.d_block(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return torch.sigmoid(out)


class DLinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(DLinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.first_conv = resnet.conv1
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.first_max_pool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.d_block = DBlockMoreDilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final_deconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = non_linearity
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = non_linearity
        self.final_conv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_max_pool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.d_block(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return torch.sigmoid(out)


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.first_conv = resnet.conv1
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.first_max_pool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final_deconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.final_relu1 = non_linearity
        self.final_conv2 = nn.Conv2d(32, 32, 3)
        self.final_relu2 = non_linearity
        self.final_conv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_max_pool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return torch.sigmoid(out)
