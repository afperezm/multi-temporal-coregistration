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
        image, label = sample['image'], sample['label']

        if 'reference' in sample:
            reference = sample['reference']
        else:
            reference = None

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

        if 'reference' in sample:
            return {'image': image, 'reference': reference, 'label': label}
        else:
            return {'image': image, 'label': label}


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
        image, label = sample['image'], sample['label']

        if 'reference' in sample:
            reference = sample['reference']
        else:
            reference = None

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
            if 'reference' in sample:
                reference = cv2.warpPerspective(reference, mat, (width, height),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=self.border_mode,
                                                borderValue=(0, 0, 0,))
            label = cv2.warpPerspective(label, mat, (width, height),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=self.border_mode,
                                        borderValue=(0, 0, 0,))

        if 'reference' in sample:
            return {'image': image, 'reference': reference, 'label': label}
        else:
            return {'image': image, 'label': label}


class RandomHorizontalFlip(object):
    """
    Horizontally flip the image and label in a sample randomly with a given probability.

    Args:
        p (float): probability of the ndarrays being flipped. Default value is 0.5
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if 'reference' in sample:
            reference = sample['reference']
        else:
            reference = None

        if np.random.random() < self.p:
            image = np.fliplr(image).copy()
            if 'reference' in sample:
                reference = np.fliplr(reference).copy()
            label = np.fliplr(label).copy()

        if 'reference' in sample:
            return {'image': image, 'reference': reference, 'label': label}
        else:
            return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    """
    Vertically flip the image and label in a sample randomly with a given probability.

    Args:
        p (float): probability of the ndarrays being flipped. Default value is 0.5
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if 'reference' in sample:
            reference = sample['reference']
        else:
            reference = None

        if np.random.random() < self.p:
            image = np.flipud(image).copy()
            if 'reference' in sample:
                reference = np.flipud(reference).copy()
            label = np.flipud(label).copy()

        if 'reference' in sample:
            return {'image': image, 'reference': reference, 'label': label}
        else:
            return {'image': image, 'label': label}


class RandomRotation(object):

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if 'reference' in sample:
            reference = sample['reference']
        else:
            reference = None

        if np.random.random() < self.p:
            image = np.rot90(image)
            if 'reference' in sample:
                reference = np.rot90(reference)
            label = np.rot90(label)

        if 'reference' in sample:
            return {'image': image, 'reference': reference, 'label': label}
        else:
            return {'image': image, 'label': label}


class Normalize(object):

    def __init__(self, feat_range=(0.0, 1.0), threshold=False):
        self.feat_range = feat_range
        self.threshold = threshold

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = np.array(image, np.float32) / 255.0
        image = image * (self.feat_range[1] - self.feat_range[0]) + self.feat_range[0]
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        label = np.array(label, np.float32) / 255.0
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=2)

        if self.threshold:
            label[label >= 0.5] = 1.0
            label[label < 0.5] = 0.0

        sample_normalized = {'image': image, 'label': label}

        if 'reference' in sample:
            reference = sample['reference']
            reference = np.array(reference, np.float32) / 255.0
            reference = reference * (self.feat_range[1] - self.feat_range[0]) + self.feat_range[0]
            sample_normalized['reference'] = reference

        return sample_normalized


class ToTensor(object):
    """Convert ndarrays in sample to tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Swap axis to place number of channels in front
        image, label = np.transpose(image, (2, 0, 1)), np.transpose(label, (2, 0, 1))

        sample_tensor = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}

        if 'reference' in sample:
            reference = sample['reference']
            reference = np.transpose(reference, (2, 0, 1))
            sample_tensor['reference'] = torch.from_numpy(reference)

        return sample_tensor
