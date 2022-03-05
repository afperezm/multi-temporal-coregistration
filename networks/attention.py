import math
import numpy as np
import torch

from torch import nn


class MultiHeadDense(nn.Module):
    def __init__(self, d):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):

        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))

        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.pos_encoder = PositionalEncoding2D(channels)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        x = self.pos_encoder(x)

        return x.permute(0, 3, 1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channels)
        self.key = MultiHeadDense(channels)
        self.value = MultiHeadDense(channels)
        self.softmax = nn.Softmax(dim=1)
        self.pos_encoder = PositionalEncodingPermute2D(channels)

    def forward(self, x):

        n, c, h, w = x.size()

        pe = self.pos_encoder(x)

        x = x + pe
        x = x.reshape(n, c, h * w).permute(0, 2, 1)  # [b, h*w, d]

        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))  # [b, h*w, h*w]
        V = self.value(x)

        x = torch.bmm(A, V).permute(0, 2, 1).reshape(n, c, h, w)

        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, channelY, channelS):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.query = MultiHeadDense(channelS)
        self.key = MultiHeadDense(channelS)
        self.value = MultiHeadDense(channelS)
        self.conv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

    def forward(self, Y, S):

        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()

        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)

        V = self.value(S1)

        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        Y2 = self.Yconv2(Y)

        Q = self.query(Y1)
        K = self.key(Y1)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Sc))

        x = torch.bmm(A, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)

        Z = self.conv(x)
        Z = Z * S
        Z = torch.cat([Z, Y2], dim=1)

        return Z


class CrossAttention(nn.Module):
    def __init__(self, num_y_channels, num_s_channels):
        super(CrossAttention, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_y_channels, num_s_channels, kernel_size=1),
            nn.BatchNorm2d(num_s_channels),
            nn.ReLU(inplace=True)
        )

        self.query = MultiHeadDense(num_s_channels)
        self.key = MultiHeadDense(num_s_channels)
        self.value = MultiHeadDense(num_s_channels)

        self.softmax = nn.Softmax(dim=1)

        self.s_pos_encoder = PositionalEncodingPermute2D(num_s_channels)
        self.y_pos_encoder = PositionalEncodingPermute2D(num_y_channels)

    def forward(self, y, s):

        sb, sc, sh, sw = s.size()

        y = torch.unsqueeze(torch.unsqueeze(y, 2), 2)
        y = y.repeat(1, 1, sh, sw)

        yb, yc, yh, yw = y.size()

        s += self.s_pos_encoder(s)
        s1 = s.reshape(yb, sc, yh * yw).permute(0, 2, 1)

        v = self.value(s1)

        y += self.y_pos_encoder(y)
        y1 = self.conv(y).reshape(yb, sc, yh * yw).permute(0, 2, 1)

        q = self.query(y1)
        k = self.key(y1)
        a = self.softmax(torch.bmm(q, k.permute(0, 2, 1)) / math.sqrt(sc))

        z = torch.bmm(a, v).permute(0, 2, 1).reshape(yb, sc, yh, yw)

        return z
