"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import sys

import cv2
import os

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from codebase.utils.transforms import RandomHSV, RandomShiftScale, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomRotation, ToTensor


class RoadsDataset(Dataset):

    def __init__(self, root, transform=None):
        super(RoadsDataset).__init__()
        image_list = list(filter(lambda x: x.find('sat') != -1, os.listdir(root)))
        train_list = list(map(lambda x: x[:-8], image_list))
        self.ids = train_list
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        index = self.ids[index]

        image = cv2.imread(os.path.join(self.root, f'{index}_sat.jpg'))
        mask = cv2.imread(os.path.join(self.root, f'{index}_mask.png'), cv2.IMREAD_GRAYSCALE)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    data_dir = sys.argv[1]

    train_dataset = RoadsDataset(root=data_dir,
                                 transform=transforms.Compose([RandomHSV(hue_shift_limit=(-30, 30),
                                                                         sat_shift_limit=(-5, 5),
                                                                         val_shift_limit=(-15, 15)),
                                                               RandomShiftScale(shift_limit=(-0.1, 0.1),
                                                                                scale_limit=(-0.1, 0.1),
                                                                                aspect_limit=(-0.1, 0.1)),
                                                               RandomHorizontalFlip(),
                                                               RandomVerticalFlip(),
                                                               RandomRotation(),
                                                               ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=4)

    for batch_idx, batch in enumerate(train_dataloader):
        print(f"batch - {batch_idx} - ", batch['image'].shape, batch['mask'].shape)
