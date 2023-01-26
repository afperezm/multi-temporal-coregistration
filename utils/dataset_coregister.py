from datetime import datetime

import argparse
import cv2
import glob
import numpy as np
import os
import pandas as pd

from enum import Enum
from sklearn.metrics import jaccard_score

MAX_TRANSLATION = 20
MAX_ROTATION = np.pi / 9


class InterpolationType(Enum):
    """Types of interpolation, available are NEAREST, LINEAR and CUBIC"""
    NEAREST = 0
    LINEAR = 1
    CUBIC = 3


def is_registration_suspicious(warp_matrix):
    """Static method that checks if estimated linear transformation could be implausible.

    This function checks whether the norm of the estimated translation or the rotation angle exceed predefined
    values. For the translation, a maximum translation radius of 20 pixels is flagged, while larger rotations than
    20 degrees are flagged.

    :param warp_matrix: Input linear transformation matrix
    :type warp_matrix: ndarray
    :return: False if registration doesn't exceed threshold, True otherwise
    """

    if warp_matrix is None:
        return True

    cos_theta = np.trace(warp_matrix[:2, :2]) / 2
    rot_angle = np.arccos(cos_theta)
    transl_norm = np.linalg.norm(warp_matrix[:, 2])

    return True if int((rot_angle > MAX_ROTATION) or (transl_norm > MAX_TRANSLATION)) else False


def register(src, trg, trg_mask=None, src_mask=None):
    """Implementation of pair-wise registration and warping using Enhanced Correlation Coefficient

    This function estimates translational transformation (x,y translation) using the intensities of the
    pair of images to be registered. The similarity metric is a modification of the cross-correlation metric, which
    is invariant to distortions in contrast and brightness.

    :param src: 2D single channel source moving image
    :param trg: 2D single channel target reference image
    :param trg_mask: Mask of target image. Not used in this method.
    :param src_mask: Mask of source image. Not used in this method.
    :return: Estimated 2D transformation matrix of shape 2x3
    """

    success = True

    # Parameters of registration
    warp_mode = cv2.MOTION_TRANSLATION

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, termination_eps)

    # Initialise warp matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        _, warp_matrix = cv2.findTransformECC(
            src.astype(np.float32),
            trg.astype(np.float32),
            warp_matrix,
            warp_mode,
            criteria,
            None,
            1,
        )
    except cv2.error:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        success = False

    # Use identity matrix if registration is suspicious
    if is_registration_suspicious(warp_matrix):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        success = False

    return warp_matrix, success


def warp(warp_matrix, img, iflag=cv2.INTER_NEAREST, interpolation_type=InterpolationType.CUBIC):
    """Function to warp input image given an estimated 2D linear transformation

    :param warp_matrix: Linear 2x3 matrix to use to linearly warp the input images
    :type warp_matrix: ndarray
    :param img: Image to be warped with estimated transformation
    :type img: ndarray
    :param iflag: Interpolation flag, specified interpolation using during resampling of warped image
    :type iflag: cv2.INTER_*
    :return: Warped image using the linear matrix
    """

    height, width = img.shape[:2]
    warped_img = np.zeros_like(img, dtype=img.dtype)
    iflag += cv2.WARP_INVERSE_MAP

    # Check if image to warp is 2D or 3D. If 3D need to loop over channels
    if (interpolation_type == InterpolationType.LINEAR) or img.ndim == 2:
        warped_img = cv2.warpAffine(img.astype(np.float32), warp_matrix, (width, height), flags=iflag).astype(
            img.dtype
        )
    elif img.ndim == 3:
        for idx in range(img.shape[-1]):
            warped_img[..., idx] = cv2.warpAffine(
                img[..., idx].astype(np.float32), warp_matrix, (width, height), flags=iflag
            ).astype(img.dtype)
    else:
        raise ValueError(f"Image has incorrect number of dimensions: {img.ndim}")

    return warped_img


def get_patch_indices(output_dir):

    masks = sorted(glob.glob(os.path.join(output_dir, f'*_mask.png')))
    patches = list(set([int('-'.join(os.path.basename(mask).split('_')[4].split('-')[1:2])) for mask in masks]))

    return sorted(patches)


def main():

    # data_dir = '/home/andresf/data/northern-cities/gillam_mb_canada_train/'
    # output_dir = '/home/andresf/workspace/DeepGlobe-Road-Extraction-Challenge/submits/dlinknet34-imagenet-gillam-all-season_train/'

    data_dir = PARAMS.data_dir
    output_dir = PARAMS.output_dir

    mean_scores_raw = []
    mean_scores_reg = []

    warp_data = []

    for eopatch_idx in get_patch_indices(output_dir):

        print(f'eopatch-{eopatch_idx:04d}')

        mask_paths = sorted(glob.glob(os.path.join(output_dir, f'*_eopatch-{eopatch_idx:04d}-{eopatch_idx:04d}_mask.png')))
        masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths]

        image_paths = sorted(glob.glob(os.path.join(data_dir, f'*_eopatch-{eopatch_idx:04d}-{eopatch_idx:04d}_sat.jpg')))
        images = [cv2.imread(image_path, cv2.IMREAD_COLOR) for image_path in image_paths]

        timestamps = [os.path.basename(mask_path).split('_')[3] for mask_path in mask_paths]
        dates = np.array([datetime.strptime(timestamp, '%Y-%m-%d-%H-%M-%S') for timestamp in timestamps])

        location = 'gillam_mb_canada'
        date = datetime(2020, 7, 1)
        closest_date_id = np.argsort(abs(date - dates))[0]
        name = f'{location}_{dates[closest_date_id].strftime("%Y-%m-%d-%H-%M-%S")}_eopatch-{eopatch_idx:04d}-{eopatch_idx:04d}_mask.png'

        print(name)

        mask_gt = cv2.imread(os.path.join(data_dir, name), cv2.IMREAD_GRAYSCALE)

        scores = [jaccard_score(mask_gt.flatten(), mask.flatten(), pos_label=255, zero_division=1) for mask in masks]
        mean_score = np.mean(scores)
        print(mean_score)
        mean_scores_raw.append(mean_score)

        mask_ref = cv2.imread(os.path.join(output_dir, name), cv2.IMREAD_GRAYSCALE)

        warp_matrices = []
        for idx, mask in enumerate(mask_paths):
            warp_matrix, succeeded = register(mask_ref, masks[idx])
            warp_data.append([os.path.splitext(os.path.basename(mask_paths[idx]))[0], succeeded, warp_matrix])
            warp_matrices.append(warp_matrix)

        masks_warped = [warp(warp_matrices[idx], mask) for idx, mask in enumerate(masks)]

        for idx, mask_path in enumerate(mask_paths):
            _ = cv2.imwrite(os.path.join(output_dir, os.path.basename(mask_path).replace('.png', '_deshifted.png')), masks_warped[idx])

        images_warped = [warp(warp_matrices[idx], image) for idx, image in enumerate(images)]

        for idx, image_path in enumerate(image_paths):
            _ = cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_deshifted.jpg')), images_warped[idx])

        scores = [jaccard_score(mask_gt.flatten(), mask.flatten(), pos_label=255, zero_division=1) for mask in masks_warped]
        mean_score = np.mean(scores)
        print(mean_score)
        mean_scores_reg.append(mean_score)

    if not os.path.exists(os.path.join(output_dir, 'warp_matrices.pkl')):
        df = pd.DataFrame(warp_data, columns=['Image', 'Success', 'Warp Matrix'])
        df.to_pickle(os.path.join(output_dir, 'warp_matrices.pkl'))

    print('mean raw score')
    print(np.mean(mean_scores_raw))
    print('mean reg score')
    print(np.mean(mean_scores_reg))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Dataset directory", required=True)
    parser.add_argument("--output_dir", help="Output directory", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
