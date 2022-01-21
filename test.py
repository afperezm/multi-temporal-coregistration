import argparse
import cv2
import numpy as np
import os
import torch

from networks.dinknet import DinkNet34
from time import time
from torch.autograd import Variable as V

BATCH_SIZE_PER_CARD = 4
PARAMS = None


class TTAFrame:
    def __init__(self, net, device):
        self.device = device
        self.net = net()
        if torch.cuda.device_count() > 0:
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.net = self.net.to(self.device)

    def test_one_img_from_path(self, path, eval_mode=True):
        if eval_mode:
            self.net.eval()
        if torch.cuda.device_count() > 0:
            batch_size = torch.cuda.device_count() * BATCH_SIZE_PER_CARD
        else:
            batch_size = BATCH_SIZE_PER_CARD
        if batch_size >= 8:
            return self.test_one_img_from_path_1(path)
        elif batch_size >= 4:
            return self.test_one_img_from_path_2(path)
        elif batch_size >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).to(device))

        mask_a = self.net.forward(img1).squeeze().cpu().data.numpy()
        mask_b = self.net.forward(img2).squeeze().cpu().data.numpy()
        mask_c = self.net.forward(img3).squeeze().cpu().data.numpy()
        mask_d = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = mask_a + mask_b[:, ::-1] + mask_c[:, :, ::-1] + mask_d[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).to(device))

        mask_a = self.net.forward(img1).squeeze().cpu().data.numpy()
        mask_b = self.net.forward(img2).squeeze().cpu().data.numpy()
        mask_c = self.net.forward(img3).squeeze().cpu().data.numpy()
        mask_d = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = mask_a + mask_b[:, ::-1] + mask_c[:, :, ::-1] + mask_d[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).to(device))
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).to(device))

        mask_a = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask_b = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = mask_a + mask_b[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).to(device))

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


def main():
    # source = 'dataset/test/'
    # source = 'dataset/valid/'
    source = PARAMS.source_dir
    # target = 'submits/log01_din34/'
    target = PARAMS.target_dir
    # weights = 'weights/log01_dink34.th'
    weights = PARAMS.checkpoint

    val = os.listdir(source)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    solver = TTAFrame(DinkNet34, dev)
    solver.load(weights)

    tic = time()

    if not os.path.exists(target):
        os.mkdir(target)

    for i, name in enumerate(val):
        if i % 10 == 0:
            print(i / 10, '    ', '%.2f' % (time() - tic))
        mask = solver.test_one_img_from_path(source + name)
        mask[mask > 4.0] = 255
        mask[mask <= 4.0] = 0
        mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
        cv2.imwrite(target + name[:-7] + 'mask.png', mask.astype(np.uint8))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", help="Source directory", required=True)
    parser.add_argument("--target_dir", help="Target directory", required=True)
    parser.add_argument("--checkpoint", help="Checkpoint file", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
