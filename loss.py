import torch
import torch.nn as nn
from pytorch_msssim import SSIM
from topoloss import get_topo_loss


class DiceBCELoss(nn.Module):
    def __init__(self, top_k=0.7):
        super(DiceBCELoss, self).__init__()
        self.top_k = top_k
        self.bce_loss = nn.BCELoss(reduction='mean') if top_k == 1 else nn.BCELoss(reduction='none')

    def soft_dice_coefficient(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.top_k == 1:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        if self.top_k == 1:
            return torch.mean(score)
        else:
            return score

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coefficient(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        if self.top_k == 1:
            bce_scores = self.bce_loss(y_pred, y_true).mean()
        else:
            bce_scores = self.bce_loss(y_pred, y_true).mean(axis=(1, 2, 3))
        dice_scores = self.soft_dice_loss(y_true, y_pred)
        scores = bce_scores + dice_scores
        if self.top_k == 1:
            return scores
        else:
            top_scores, top_indices = torch.topk(scores, int(self.top_k * scores.size()[0]))
            return torch.mean(top_scores)


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
        topo_loss_value = torch.stack([get_topo_loss(prediction, label, self.topo_size) for prediction, label in
                                       zip(torch.unbind(predictions, dim=0), torch.unbind(labels, dim=0))],
                                      dim=0).mean()

        return bce_loss_value + self.topo_weight * topo_loss_value
