"""
Based on https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge
"""
import cv2
import os
import sys

from codebase.utils import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose


class RoadsDataset(Dataset):

    train_phase = 'train'
    test_phase = 'test'

    def __init__(self, data_dir, is_train=True, transform=None):
        super(RoadsDataset).__init__()

        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform

        if self.is_train:
            self.phase = self.train_phase
        else:
            self.phase = self.test_phase

        image_list = list(filter(lambda x: x.find('sat') != -1, os.listdir(os.path.join(data_dir, self.phase))))
        train_list = list(map(lambda x: x[:-8], image_list))

        self.ids = train_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        index = self.ids[index]

        image = cv2.imread(os.path.join(self.data_dir, self.phase, f'{index}_sat.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.data_dir, self.phase, f'{index}_mask.png'), cv2.IMREAD_GRAYSCALE)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    data_dir = sys.argv[1]

    train_dataset = RoadsDataset(data_dir=data_dir,
                                 is_train=True,
                                 transform=Compose([transforms.Normalize(), transforms.ToTensor()]))
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)

    for batch_idx, batch in enumerate(train_dataloader):
        print(f"batch - {batch_idx} - ", batch[0].shape, batch[1].shape)
