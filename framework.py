import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np


class MyFrame:
    def __init__(self, net, loss, metric, device, lr=2e-4, eval_mode=False):
        self.device = device
        self.net = net()
        if torch.cuda.device_count() > 0:
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.net = self.net.to(self.device)
        # encoder_params = {*self.net.first_conv.parameters(),
        #                   *self.net.first_bn.parameters(),
        #                   *self.net.first_relu.parameters(),
        #                   *self.net.first_max_pool.parameters(),
        #                   *self.net.encoder1.parameters(),
        #                   *self.net.encoder2.parameters(),
        #                   *self.net.encoder3.parameters(),
        #                   *self.net.encoder4.parameters(),
        #                   *self.net.encoder6.heads.parameters()}
        # self.optimizer = torch.optim.Adam(params=set(self.net.parameters()).difference(encoder_params), lr=lr)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.metric = metric()
        self.old_lr = lr
        if eval_mode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).to(self.device))

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.to(self.device), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.to(self.device), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(pred, self.mask)
        accuracy = self.metric(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item(), accuracy.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, my_log, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=my_log)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
