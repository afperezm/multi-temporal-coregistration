import torch.nn as nn
from pytorch_msssim import SSIM
from soft_skeleton import soft_skel
from topoloss import get_topo_loss


class DiceBCELoss(nn.Module):
    def __init__(self, batch=True):
        super(DiceBCELoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coefficient(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coefficient(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim_module = SSIM(data_range=1.0, size_average=True, win_size=3, win_sigma=1.5, channel=1)

    def forward(self, labels, predictions):
        ssim_value = self.ssim_module(predictions, labels)
        jac_loss_value = 1 - ssim_value
        return jac_loss_value


class ComboTopoLoss(nn.Module):
    def __init__(self, topo_size=128, topo_weight=1.0):
        super(ComboTopoLoss, self).__init__()
        self.topo_size = topo_size
        self.topo_weight = topo_weight

    def forward(self, labels, predictions):

        bce_loss_value = torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels)

        labels = 1 - labels
        predictions = 1 - predictions
        topo_loss_value = torch.stack(
            [get_topo_loss(prediction.squeeze(), label.squeeze(), self.topo_size) for prediction, label in
             zip(torch.unbind(predictions, dim=0), torch.unbind(labels, dim=0))], dim=0).mean()

        return bce_loss_value + self.topo_weight * topo_loss_value


class SoftClDice(nn.Module):
    def __init__(self, iter_=3, smooth=1.):
        super(SoftClDice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        iters_ = self.iter
        smooth = self.smooth

        skel_pred = soft_skel(y_pred, iters_)
        skel_true = soft_skel(y_true, iters_)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 0:, ...]) + smooth) / (
                torch.sum(skel_pred[:, 0:, ...]) + smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 0:, ...]) + smooth) / (
                torch.sum(skel_true[:, 0:, ...]) + smooth)

        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)

        return cl_dice


class SoftDiceClDice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1.):
        super(SoftDiceClDice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def soft_dice(self, y_true, y_pred, smooth=1):
        """[function to compute dice loss]

        Args:
            y_true ([float32]): [ground truth image]
            y_pred ([float32]): [predicted image]

        Returns:
            [float32]: [loss value]
        """

        intersection = torch.sum((y_true * y_pred)[:, 0:, ...])
        coefficient = (2. * intersection + smooth) / (
                torch.sum(y_true[:, 0:, ...]) + torch.sum(y_pred[:, 0:, ...]) + smooth)

        return 1. - coefficient

    def forward(self, y_true, y_pred):

        dice = self.soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 0:, ...]) + self.smooth) / (
                torch.sum(skel_pred[:, 0:, ...]) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 0:, ...]) + self.smooth) / (
                torch.sum(skel_true[:, 0:, ...]) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)

        loss = (1.0 - self.alpha) * dice + self.alpha * cl_dice
        loss.requires_grad = True

        return loss


if __name__ == "__main__":

    import cv2
    import numpy as np
    import torch

    # from loss import ComboTopoLoss

    data_dir = '/Users/perezmaf/data/northern-cities/test_gillam_nonempty/'
    name = 'gillam_mb_canada_2020-07-29-17-38-19_eopatch-0019-0019_sat.jpg'

    mask = cv2.imread(data_dir + name.replace('_sat.jpg', '_mask.png'), cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask, np.float32) / 255.0
    mask[mask >= 0.5] = 1.0
    mask[mask <= 0.5] = 0.0
    mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)

    pred_name = 'gillam_mb_canada_2020-07-29-17-38-19_eopatch-0019-0019_sat.jpg.npy'
    pred = np.load(pred_name)
    pred = np.expand_dims(np.expand_dims(pred, axis=0), axis=0)

    loss = ComboTopoLoss(64)
    bce_topo_loss = loss(torch.tensor(mask), torch.tensor(pred))
    print(bce_topo_loss)
