import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.down1 = self.conv_stage(3, 8)
        self.down2 = self.conv_stage(8, 16)
        self.down3 = self.conv_stage(16, 32)
        self.down4 = self.conv_stage(32, 64)
        self.down5 = self.conv_stage(64, 128)
        self.down6 = self.conv_stage(128, 256)
        self.down7 = self.conv_stage(256, 512)

        self.center = self.conv_stage(512, 1024)
        # self.center_res = self.res_block(1024)

        self.up7 = self.conv_stage(1024, 512)
        self.up6 = self.conv_stage(512, 256)
        self.up5 = self.conv_stage(256, 128)
        self.up4 = self.conv_stage(128, 64)
        self.up3 = self.conv_stage(64, 32)
        self.up2 = self.conv_stage(32, 16)
        self.up1 = self.conv_stage(16, 8)

        self.trans7 = self.up_sample(1024, 512)
        self.trans6 = self.up_sample(512, 256)
        self.trans5 = self.up_sample(256, 128)
        self.trans4 = self.up_sample(128, 64)
        self.trans3 = self.up_sample(64, 32)
        self.trans2 = self.up_sample(32, 16)
        self.trans1 = self.up_sample(16, 8)

        self.conv_last = nn.Sequential(
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
        if useBN:
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(dim_out),
                # nn.LeakyReLU(0.1),
                nn.ReLU(),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(dim_out),
                # nn.LeakyReLU(0.1),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU()
            )

    def up_sample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        conv1_out = self.down1(x)
        conv2_out = self.down2(self.max_pool(conv1_out))
        conv3_out = self.down3(self.max_pool(conv2_out))
        conv4_out = self.down4(self.max_pool(conv3_out))
        conv5_out = self.down5(self.max_pool(conv4_out))
        conv6_out = self.down6(self.max_pool(conv5_out))
        conv7_out = self.down7(self.max_pool(conv6_out))

        out = self.center(self.max_pool(conv7_out))
        # out = self.center_res(out)

        out = self.up7(torch.cat((self.trans7(out), conv7_out), 1))
        out = self.up6(torch.cat((self.trans6(out), conv6_out), 1))
        out = self.up5(torch.cat((self.trans5(out), conv5_out), 1))
        out = self.up4(torch.cat((self.trans4(out), conv4_out), 1))
        out = self.up3(torch.cat((self.trans3(out), conv3_out), 1))
        out = self.up2(torch.cat((self.trans2(out), conv2_out), 1))
        out = self.up1(torch.cat((self.trans1(out), conv1_out), 1))

        out = self.conv_last(out)

        return out


class ResNetUNet(nn.Module):
    def __init__(self, feature_indices=(0, 4, 5, 6, 7), feature_channels=(64, 64, 128, 256, 512)):
        """
        Creates a UNet from a pretrained backbone.
        """
        super(ResNetUNet, self).__init__()

        backbone = models.resnet18(pretrained=True)

        encoder = SegmentationEncoder(backbone, feature_indices, feature_channels, diff=True)
        self.model = UNet(encoder, feature_channels, 1, bilinear=True, concat_mult=1, dropout_rate=0.3)

    def forward(self, x):
        out = self.model.forward(x)
        return torch.sigmoid(out)


class SegmentationEncoder(nn.Module):
    def __init__(self, backbone, feature_indices, feature_channels, diff=False):
        super(SegmentationEncoder, self).__init__()
        self.feature_indices = list(sorted(feature_indices))

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels = feature_channels  # [3, 16, 64, 128, 256, 512]

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels = 3

        # Define encoder modules below
        self.encoder = backbone

        self.diff = diff

    def forward(self, x1, x2):
        """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
        """
        feats = [self.concatenate(x1, x2)]
        for i, module in enumerate(self.encoder.children()):
            x1 = module(x1)
            x2 = module(x2)
            if i in self.feature_indices:
                feats.append(self.concatenate(x1, x2))
            if i == self.feature_indices[-1]:
                break

        return feats

    def concatenate(self, x1, x2):
        if self.diff:
            return torch.abs(x1 - x2)
        else:
            torch.cat([x1, x2], 1)


class UNet(nn.Module):
    def __init__(self, encoder, feature_channels, n_classes, concat_mult=2, bilinear=True, dropout_rate=0.5):
        """Simple segmentation network

        Args:
            encoder (torch Sequential): The pre-trained encoder
            feature_channels (list(int)): Number of channels per input feature
            n_classes (int): output number of classes
            concat_mult (int, optional): The amount of features being fused. Defaults to 2.
            bilinear (bool, optional): If use bilinear interpolation (I have defaulted to nearest since it has been shown to be better sometimes). Defaults to True.
        """
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.feature_channels = feature_channels
        self.dropout = torch.nn.Dropout2d(dropout_rate)
        for i in range(0, len(feature_channels) - 1):
            if i == len(feature_channels) - 2:
                in_ch = feature_channels[i + 1] * concat_mult
            else:
                in_ch = feature_channels[i + 1] * concat_mult
            setattr(self, 'shrink%d' % i,
                    nn.Conv2d(in_ch, feature_channels[i] * concat_mult, kernel_size=3, stride=1, padding=1))
            setattr(self, 'shrink2%d' % i,
                    nn.Conv2d(feature_channels[i] * concat_mult * 2, feature_channels[i] * concat_mult, kernel_size=3, stride=1, padding=1, bias=False))
            setattr(self, 'batchnorm%d' % i,
                    nn.BatchNorm2d(feature_channels[i] * concat_mult))
        self.outc = OutConv(feature_channels[0] * concat_mult, n_classes)
        self.encoder = encoder

    def forward(self, *in_x):
        features = self.encoder(*in_x)
        features = features[1:]
        x = features[-1]
        for i in range(len(features) - 2, -1, -1):
            conv = getattr(self, 'shrink%d' % i)
            x = F.upsample_nearest(x, scale_factor=2)
            x = conv(x)
            if features[i].shape[-1] != x.shape[-1]:
                x2 = F.upsample_nearest(features[i], scale_factor=2)
            else:
                x2 = features[i]
            x = torch.cat([x, x2], 1)
            conv2 = getattr(self, 'shrink2%d' % i)
            x = conv2(x)
            bn = getattr(self, 'batchnorm%d' % i)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = F.upsample_nearest(x, scale_factor=2)
        logits = self.outc(x)
        return logits


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
