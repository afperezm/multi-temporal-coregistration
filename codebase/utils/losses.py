# import cv2
# import malis as m
# import random
# import numpy as np
# import time
import torch
import torch.nn as nn

# from pytorch_msssim import SSIM
# from skimage import measure
# from soft_skeleton import soft_skel
# from topoloss import get_topo_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def soft_dice_coefficient(self, y_pred, y_true):
        total = torch.sum(y_pred) + torch.sum(y_true)
        intersection = torch.sum(y_pred * y_true)
        score = (2. * intersection + self.smooth) / (total + self.smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        dice_value = self.soft_dice_coefficient(y_pred, y_true)
        return 1 - dice_value

    def __call__(self, y_pred, y_true):
        b = self.soft_dice_loss(y_pred, y_true)
        return b


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=0.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce_loss = nn.BCELoss()
        self.soft_dice_loss = DiceLoss(self.smooth)

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)
        return a + b


# class SSIMLoss(nn.Module):
#     def __init__(self):
#         super(SSIMLoss, self).__init__()
#         self.ssim_module = SSIM(data_range=1.0, size_average=True, win_size=3, win_sigma=1.5, channel=1)
#
#     def forward(self, labels, predictions):
#         ssim_value = self.ssim_module(predictions, labels)
#         jac_loss_value = 1 - ssim_value
#         return jac_loss_value


# class PersistenceLoss(nn.Module):
#     def __init__(self, topo_size=64, topo_weight=0.005, smooth=1e-7):
#         super(PersistenceLoss, self).__init__()
#         self.topo_size = topo_size
#         self.topo_weight = topo_weight
#         self.smooth = smooth
#         self.bce_loss = nn.BCELoss()
#
#     def soft_dice_loss(self, y_true, y_pred):
#
#         i = torch.sum(y_true)
#         j = torch.sum(y_pred)
#
#         intersection = torch.sum(y_true * y_pred)
#         score = (2. * intersection + self.smooth) / (i + j + self.smooth)
#
#         loss_value = 1 - score.mean()
#
#         return loss_value
#
#     def topo_loss(self, y_true, y_pred):
#
#         topo_scores = [get_topo_loss(prediction.squeeze(), label.squeeze(), self.topo_size) for prediction, label in
#                        zip(torch.unbind(y_pred, dim=0), torch.unbind(y_true, dim=0))]
#         topo_loss_value = torch.stack(topo_scores, dim=0).mean()
#
#         return topo_loss_value
#
#     def forward(self, y_true, y_pred):
#
#         bce_loss_value = self.bce_loss(y_pred, y_true)
#         dice_loss_value = self.soft_dice_loss(y_true, y_pred)
#         topo_loss_value = self.topo_loss(y_true, y_pred)
#
#         return bce_loss_value + dice_loss_value + self.topo_weight * topo_loss_value


# class SoftCenterlineDiceLoss(nn.Module):
#     def __init__(self, iter_=3, smooth=1.):
#         super(SoftCenterlineDiceLoss, self).__init__()
#         self.iter = iter_
#         self.smooth = smooth
#
#     def forward(self, y_true, y_pred):
#         iters_ = self.iter
#         smooth = self.smooth
#
#         skel_pred = soft_skel(y_pred, iters_)
#         skel_true = soft_skel(y_true, iters_)
#         tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 0:, ...]) + smooth) / (
#                 torch.sum(skel_pred[:, 0:, ...]) + smooth)
#         tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 0:, ...]) + smooth) / (
#                 torch.sum(skel_true[:, 0:, ...]) + smooth)
#
#         cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
#
#         return cl_dice


# class SoftDiceClDiceLoss(nn.Module):
#     def __init__(self, iterations=20, alpha=0.5, smooth=1e-7):
#         super(SoftDiceClDiceLoss, self).__init__()
#         self.iterations = iterations
#         self.smooth = smooth
#         self.alpha = alpha
#
#     def soft_dice(self, y_true, y_pred):
#         """[function to compute dice loss]
#
#         Args:
#             y_true ([float32]): [ground truth image]
#             y_pred ([float32]): [predicted image]
#
#         Returns:
#             [float32]: [loss value]
#         """
#
#         intersection = torch.sum((y_true * y_pred)[:, 0:, ...])
#         coefficient = (2. * intersection + self.smooth) / (
#                 torch.sum(y_true[:, 0:, ...]) + torch.sum(y_pred[:, 0:, ...]) + self.smooth)
#
#         return 1. - coefficient
#
#     def forward(self, y_true, y_pred):
#
#         dice = self.soft_dice(y_true, y_pred)
#         skel_pred = soft_skel(y_pred, self.iterations)
#         skel_true = soft_skel(y_true, self.iterations)
#         tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 0:, ...]) + self.smooth) / (
#                 torch.sum(skel_pred[:, 0:, ...]) + self.smooth)
#         tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 0:, ...]) + self.smooth) / (
#                 torch.sum(skel_true[:, 0:, ...]) + self.smooth)
#         cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
#
#         loss_value = (1.0 - self.alpha) * dice + self.alpha * cl_dice
#
#         return loss_value


# class ConnectivityLoss(nn.Module):
#
#     def __init__(self, ignore_index=255, window=32, malis_lr=1.0, malis_lr_pos=1.0):
#         super().__init__()
#         self.ignore_index = ignore_index
#         self.window = window
#         self.malis_lr = malis_lr
#         self.malis_lr_pos = malis_lr_pos
#         self.nodes_indexes = self.create_nodes_indexes()
#
#     def create_nodes_indexes(self):
#         nodes_indexes = torch.arange(self.window * self.window).reshape(self.window, self.window)
#         nodes_indexes_h = torch.vstack([nodes_indexes[:, :-1].flatten(), nodes_indexes[:, 1:].flatten()])
#         nodes_indexes_v = torch.vstack([nodes_indexes[:-1, :].flatten(), nodes_indexes[1:, :].flatten()])
#
#         return torch.hstack([nodes_indexes_h, nodes_indexes_v])
#
#     def forward(self, y_true, y_pred):
#
#         B, C, H, W = y_pred.shape
#
#         weights_n = torch.zeros_like(y_pred, dtype=torch.float64).to(y_pred.device)
#         weights_p = torch.zeros_like(y_pred, dtype=torch.float64).to(y_pred.device)
#
#         for row_idx in range(H // self.window):
#             for col_idx in range(W // self.window):
#
#                 pred_np = y_pred[:, :, row_idx * self.window:(row_idx + 1) * self.window, col_idx * self.window:(col_idx + 1) * self.window]
#                 target_np = y_true[:, :, row_idx * self.window:(row_idx + 1) * self.window, col_idx * self.window:(col_idx + 1) * self.window]
#
#                 if torch.min(pred_np) == 1 or torch.max(pred_np) == 0:
#                     continue
#                 if torch.min(target_np) == 1 or torch.max(target_np) == 0:
#                     continue
#
#                 costs_h = (pred_np[:, :, :, :-1] + pred_np[:, :, :, 1:]).reshape(B, -1)
#                 costs_v = (pred_np[:, :, :-1, :] + pred_np[:, :, 1:, :]).reshape(B, -1)
#                 costs = torch.hstack([costs_h, costs_v])
#
#                 gt_costs_h = (target_np[:, :, :, :-1] + target_np[:, :, :, 1:]).reshape(B, -1)
#                 gt_costs_v = (target_np[:, :, :-1, :] + target_np[:, :, 1:, :]).reshape(B, -1)
#                 gt_costs = torch.hstack([gt_costs_h, gt_costs_v])
#
#                 costs_n = costs.clone()
#                 costs_p = costs.clone()
#
#                 costs_n[gt_costs > 20] = 20
#                 costs_p[gt_costs < 10] = 0
#                 gt_costs[gt_costs > 20] = 20
#
#                 for i in range(len(pred_np)):
#                     sg_gt = measure.label(target_np[i, 0].cpu().detach().numpy() == 0)
#
#                     edge_weights_n = m.malis_loss_weights(sg_gt.astype(np.uint64).flatten(),
#                                                           self.nodes_indexes[0].cpu().detach().numpy().astype(np.uint64),
#                                                           self.nodes_indexes[1].cpu().detach().numpy().astype(np.uint64),
#                                                           costs_n[i].cpu().detach().numpy().astype(np.float32), 0)
#                     edge_weights_n = torch.tensor(edge_weights_n.astype(np.int64)).to(y_pred.device)
#
#                     edge_weights_p = m.malis_loss_weights(sg_gt.astype(np.uint64).flatten(),
#                                                           self.nodes_indexes[0].cpu().detach().numpy().astype(np.uint64),
#                                                           self.nodes_indexes[1].cpu().detach().numpy().astype(np.uint64),
#                                                           costs_p[i].cpu().detach().numpy().astype(np.float32), 1)
#                     edge_weights_p = torch.tensor(edge_weights_p.astype(np.int64)).to(y_pred.device)
#
#                     num_pairs_n = torch.sum(edge_weights_n)
#                     if num_pairs_n > 0:
#                         edge_weights_n = edge_weights_n / num_pairs_n
#
#                     num_pairs_p = torch.sum(edge_weights_p)
#                     if num_pairs_p > 0:
#                         edge_weights_p = edge_weights_p / num_pairs_p
#
#                     # Depending on your clip values
#                     edge_weights_n[gt_costs[i] >= 10] = 0
#                     edge_weights_p[gt_costs[i] < 20] = 0
#
#                     malis_w = edge_weights_n.clone()
#
#                     malis_w_h, malis_w_v = torch.split(malis_w, torch.numel(malis_w) // 2)
#                     malis_w_h, malis_w_v = malis_w_h.reshape(self.window, self.window - 1), malis_w_v.reshape(self.window - 1, self.window)
#
#                     nodes_weights = weights_n[i, 0, row_idx * self.window:(row_idx + 1) * self.window, col_idx * self.window:(col_idx + 1) * self.window]
#                     nodes_weights[:, :-1] += malis_w_h
#                     nodes_weights[:, 1:] += malis_w_h
#                     nodes_weights[:-1, :] += malis_w_v
#                     nodes_weights[1:, :] += malis_w_v
#
#                     malis_w = edge_weights_p.clone()
#
#                     malis_w_h, malis_w_v = torch.split(malis_w, torch.numel(malis_w) // 2)
#                     malis_w_h, malis_w_v = malis_w_h.reshape(self.window, self.window - 1), malis_w_v.reshape( self.window - 1, self.window)
#
#                     nodes_weights = weights_p[i, 0, row_idx * self.window:(row_idx + 1) * self.window, col_idx * self.window:(col_idx + 1) * self.window]
#                     nodes_weights[:, :-1] += malis_w_h
#                     nodes_weights[:, 1:] += malis_w_h
#                     nodes_weights[:-1, :] += malis_w_v
#                     nodes_weights[1:, :] += malis_w_v
#
#         loss_n = y_pred.pow(2)
#         loss_p = (20 - y_pred).pow(2)
#
#         loss_value = self.malis_lr * loss_n * weights_n + self.malis_lr_pos * loss_p * weights_p
#
#         return loss_value.sum()


# def apply_edt(img):
#     result = cv2.distanceTransform(img.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
#     result2 = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     return result2


# if __name__ == "__main__":
#
#     data_dir = '/home/andresf/data/deepglobe-roads/train/'
#     name = '768086_sat.jpg'
#
#     mask = cv2.imread(data_dir + name.replace('_sat.jpg', '_mask.png'), cv2.IMREAD_GRAYSCALE)
#     mask = np.array(mask, np.float32) / 255.0
#     mask[mask >= 0.5] = 1.0
#     mask[mask <= 0.5] = 0.0
#     # mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)
#
#     pred_name = '/home/andresf/Downloads/768086_mask.npy'
#     pred = np.load(pred_name)
#     # pred = np.array(pred, np.float32) / 8.0
#     # pred = np.expand_dims(np.expand_dims(pred, axis=0), axis=0)
#     pred[pred > 4.0] = 255
#     pred[pred <= 4.0] = 0
#
#     # loss = PersistenceLoss()
#     #
#     # random.seed(42)
#     #
#     # t0 = time.time()
#     # topo_loss = loss(torch.tensor(mask), torch.tensor(pred))
#     # t1 = time.time()
#     #
#     # print(t1 - t0, topo_loss)
#
#     mask_edt = apply_edt(mask)
#     pred_edt = apply_edt(pred)
#
#     from matplotlib import pyplot as plt
#
#     fig, axes = plt.subplots(nrows=2, ncols=2)
#
#     axes[0][0].imshow(mask)
#     axes[0][1].imshow(pred)
#     axes[1][0].imshow(mask_edt, cmap=plt.get_cmap('gray'))
#     axes[1][1].imshow(pred_edt, cmap=plt.get_cmap('gray'))
#
#     fig.show()
#
#     mask_edt = np.expand_dims(np.expand_dims(mask_edt, axis=0), axis=0)
#     pred_edt = np.expand_dims(np.expand_dims(pred_edt, axis=0), axis=0)
#
#     mask_edt = 1 - mask_edt.astype(np.float32) / 255.0
#     pred_edt = 1 - pred_edt.astype(np.float32) / 255.0
#
#     # loss = nn.MSELoss()
#     loss = ConnectivityLoss()
#
#     t0 = time.time()
#     topo_loss = loss(torch.tensor(mask_edt), torch.tensor(pred_edt))
#     # mse_loss = loss(torch.tensor(pred_edt), torch.tensor(mask_edt))
#     t1 = time.time()
#
#     print(t1 - t0, topo_loss)
