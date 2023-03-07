import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=0.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce_loss = nn.BCELoss()

    def soft_dice_coefficient(self, y_pred, y_true):
        total = torch.sum(y_pred) + torch.sum(y_true)
        intersection = torch.sum(y_pred * y_true)
        score = (2. * intersection + self.smooth) / (total + self.smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        dice_value = self.soft_dice_coefficient(y_pred, y_true)
        return 1 - dice_value

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)
        return a + b
