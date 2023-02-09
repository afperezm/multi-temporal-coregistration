import argparse
import cv2
import glob
import numpy as np
import os
import pandas as pd

from skimage.metrics import structural_similarity as ssim

PARAMS = None


def normalize(array: np.ndarray) -> np.ndarray:
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val == min_val:
        max_val += 1e-5
    return ((array - min_val) / (max_val - min_val)).astype(np.float64)


def get_patch_indices(output_dir):

    images = sorted(glob.glob(os.path.join(output_dir, f'*.jpg')))
    patches = list(set([int('-'.join(os.path.basename(image).split('_')[4].split('-')[1:2])) for image in images]))

    return sorted(patches)


def main():

    submits_dir = PARAMS.submits_dir
    band = PARAMS.band

    data_dirs = sorted(glob.glob(os.path.join(submits_dir, 'dlinknet34-imagenet-gillam-all-season-fold-000[1-5]-[0-9]*-[0-9]*')))

    folds = []
    ssim_scores = []

    for data_dir in data_dirs:

        fold = int(os.path.basename(os.path.split(data_dir)[0]).split('-')[6])

        print(f'fold={fold}')

        scores = []

        for eopatch_idx in get_patch_indices(data_dir):

            print(f'eopatch-{eopatch_idx:04d}')

            image_paths = sorted(glob.glob(os.path.join(data_dir, f'*_eopatch-{eopatch_idx:04d}-{eopatch_idx:04d}_*.jpg')))

            for tgt_path, ref_path in zip(image_paths[-1:] + image_paths[:-1], image_paths):
                ref = cv2.imread(ref_path)
                tgt = cv2.imread(tgt_path)
                score = ssim(normalize(ref[:, :, band]), normalize(tgt[:, :, band]), data_range=1)
                scores.append(score)

        folds.append(fold)
        ssim_scores.append(np.mean(scores))

    if not os.path.exists(os.path.join(submits_dir, 'ssim_scores.pkl')):
        df = pd.DataFrame(list(zip(folds, ssim_scores)), columns=['Fold', 'SSIM'])
        df.to_pickle(os.path.join(submits_dir, 'ssim_scores.pkl'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submits_dir", help="Submits directory", required=True)
    parser.add_argument("--band", help="Band used for measuring structural similarity", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
