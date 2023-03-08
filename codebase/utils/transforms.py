import cv2
import numpy as np
import torch


class RandomHSV(object):

    def __init__(self, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), p=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.p:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.randint(self.hue_shift_limit[0], self.hue_shift_limit[1] + 1)
            hue_shift = np.uint8(hue_shift)
            h += hue_shift
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            # image = cv2.merge((s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return {'image': image, 'mask': mask}


class RandomShiftScale(object):

    def __init__(self, shift_limit=(-0.0, 0.0), scale_limit=(-0.0, 0.0), aspect_limit=(-0.0, 0.0),
                 borderMode=cv2.BORDER_CONSTANT, p=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = (-0.0, 0.0)
        self.aspect_limit = aspect_limit
        self.border_mode = borderMode
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.p:
            height, width, channel = image.shape

            angle = np.random.uniform(self.rotate_limit[0], self.rotate_limit[1])
            scale = np.random.uniform(1 + self.scale_limit[0], 1 + self.scale_limit[1])
            aspect = np.random.uniform(1 + self.aspect_limit[0], 1 + self.aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * width)
            dy = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=self.border_mode,
                                        borderValue=(0, 0, 0,))
            mask = cv2.warpPerspective(mask, mat, (width, height),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=self.border_mode,
                                       borderValue=(0, 0, 0,))

        return {'image': image, 'mask': mask}


class RandomHorizontalFlip(object):

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return {'image': image, 'mask': mask}


class RandomVerticalFlip(object):

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.p:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        return {'image': image, 'mask': mask}


class RandomRotation(object):

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.p:
            image = np.rot90(image)
            mask = np.rot90(mask)

        return {'image': image, 'mask': mask}


class Normalize(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        mask = np.expand_dims(mask, axis=2)

        image = np.array(image, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose((2, 0, 1)) / 255.0

        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

        return {'image': image, 'mask': mask}


class ToTensor(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']



        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}
