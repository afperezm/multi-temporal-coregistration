import malis as m
import torch.nn as nn

from pytorch_msssim import SSIM
from skimage import measure
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


class PersistenceLoss(nn.Module):
    def __init__(self, topo_size=128, topo_weight=1.0):
        super(PersistenceLoss, self).__init__()
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


class SoftCenterlineDiceLoss(nn.Module):
    def __init__(self, iter_=3, smooth=1.):
        super(SoftCenterlineDiceLoss, self).__init__()
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


class SoftDiceCenterlineDiceLoss(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1.):
        super(SoftDiceCenterlineDiceLoss, self).__init__()
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


class ConnectivityLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, y_true, y_pred, malis_lr=1.0, malis_lr_pos=1.0):

        pred_np_full = y_pred
        target_np_full = y_true

        B, C, H, W = pred_np_full.shape

        weights_n = torch.zeros_like(pred_np_full, dtype=torch.float64).to(y_pred.device)
        weights_p = torch.zeros_like(pred_np_full, dtype=torch.float64).to(y_pred.device)

        window = 32

        for row_idx in range(H // window):
            for col_idx in range(W // window):
                pred_np = pred_np_full[:, :, row_idx * window:(row_idx + 1) * window, col_idx * window:(col_idx + 1) * window]
                target_np = target_np_full[:, :, row_idx * window:(row_idx + 1) * window, col_idx * window:(col_idx + 1) * window]

                nodes_indexes = torch.arange(window * window).reshape(window, window)
                nodes_indexes_h = torch.vstack([nodes_indexes[:, :-1].flatten(), nodes_indexes[:, 1:].flatten()])
                nodes_indexes_v = torch.vstack([nodes_indexes[:-1, :].flatten(), nodes_indexes[1:, :].flatten()])
                nodes_indexes = torch.hstack([nodes_indexes_h, nodes_indexes_v])

                costs_h = (pred_np[:, :, :, :-1] + pred_np[:, :, :, 1:]).reshape(B, -1)
                costs_v = (pred_np[:, :, :-1, :] + pred_np[:, :, 1:, :]).reshape(B, -1)
                costs = torch.hstack([costs_h, costs_v])

                gt_costs_h = (target_np[:, :, :, :-1] + target_np[:, :, :, 1:]).reshape(B, -1)
                gt_costs_v = (target_np[:, :, :-1, :] + target_np[:, :, 1:, :]).reshape(B, -1)
                gt_costs = torch.hstack([gt_costs_h, gt_costs_v])

                costs_n = costs.clone()
                costs_p = costs.clone()

                costs_n[gt_costs > 20] = 20
                costs_p[gt_costs < 10] = 0
                gt_costs[gt_costs > 20] = 20

                for i in range(len(pred_np)):
                    sg_gt = measure.label(target_np[i, 0] == 0)

                    edge_weights_n = m.malis_loss_weights(sg_gt.astype(np.uint64).flatten(), nodes_indexes[0].cpu().detach().numpy().astype(np.uint64),
                                                          nodes_indexes[1].cpu().detach().numpy().astype(np.uint64), costs_n[i].cpu().detach().numpy().astype(np.float32), 0)
                    edge_weights_n = torch.tensor(edge_weights_n.astype(np.int64)).to(y_pred.device)

                    edge_weights_p = m.malis_loss_weights(sg_gt.astype(np.uint64).flatten(), nodes_indexes[0].cpu().detach().numpy().astype(np.uint64),
                                                          nodes_indexes[1].cpu().detach().numpy().astype(np.uint64), costs_p[i].cpu().detach().numpy().astype(np.float32), 1)
                    edge_weights_p = torch.tensor(edge_weights_p.astype(np.int64)).to(y_pred.device)

                    num_pairs_n = torch.sum(edge_weights_n)
                    if num_pairs_n > 0:
                        edge_weights_n = edge_weights_n / num_pairs_n

                    num_pairs_p = torch.sum(edge_weights_p)
                    if num_pairs_p > 0:
                        edge_weights_p = edge_weights_p / num_pairs_p

                    # Depending on your clip values
                    edge_weights_n[gt_costs[i] >= 10] = 0
                    edge_weights_p[gt_costs[i] < 20] = 0

                    malis_w = edge_weights_n.clone()

                    malis_w_h, malis_w_v = torch.split(malis_w, torch.numel(malis_w) // 2)
                    malis_w_h, malis_w_v = malis_w_h.reshape(window, window - 1), malis_w_v.reshape(window - 1, window)

                    nodes_weights = torch.zeros(window, window, dtype=torch.float64).to(y_pred.device)
                    nodes_weights[:, :-1] += malis_w_h
                    nodes_weights[:, 1:] += malis_w_h
                    nodes_weights[:-1, :] += malis_w_v
                    nodes_weights[1:, :] += malis_w_v

                    weights_n[i, 0, row_idx * window:(row_idx + 1) * window, col_idx * window:(col_idx + 1) * window] = nodes_weights

                    malis_w = edge_weights_p.clone()

                    malis_w_h, malis_w_v = torch.split(malis_w, torch.numel(malis_w) // 2)
                    malis_w_h, malis_w_v = malis_w_h.reshape(window, window - 1), malis_w_v.reshape(window - 1, window)

                    nodes_weights = torch.zeros(window, window, dtype=torch.float32).to(y_pred.device)
                    nodes_weights[:, :-1] += malis_w_h
                    nodes_weights[:, 1:] += malis_w_h
                    nodes_weights[:-1, :] += malis_w_v
                    nodes_weights[1:, :] += malis_w_v

                    weights_p[i, 0, row_idx * window:(row_idx + 1) * window, col_idx * window:(col_idx + 1) * window] = nodes_weights

        loss_n = y_pred.pow(2)
        loss_p = (20 - y_pred).pow(2)

        loss_value = malis_lr * loss_n * weights_n + malis_lr_pos * loss_p * weights_p

        return loss_value.sum()


if __name__ == "__main__":
    import cv2
    import numpy as np
    import torch

    # from loss import ComboTopoLoss

    data_dir = '/home/andresf/data/northern-cities/test/'
    name = 'gillam_mb_canada_2020-07-29-17-38-19_eopatch-0019-0019_sat.jpg'

    mask = cv2.imread(data_dir + name.replace('_sat.jpg', '_mask.png'), cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask, np.float32) / 255.0
    mask[mask >= 0.5] = 1.0
    mask[mask <= 0.5] = 0.0
    mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)

    pred_name = 'gillam_mb_canada_2020-07-29-17-38-19_eopatch-0019-0019_sat.jpg.npy'
    pred = np.load(pred_name)
    pred = np.expand_dims(np.expand_dims(pred, axis=0), axis=0)

    loss = ConnectivityLoss()
    topo_loss = loss(torch.tensor(mask), torch.tensor(pred), 1.0, 1.0)
    print(topo_loss)
