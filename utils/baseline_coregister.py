import argparse
import glob
import tempfile

import numpy as np
import os
import pandas as pd
import shutil
import subprocess
import warnings

from arosics import COREG
from datetime import datetime

PARAMS = None


def get_patch_indices(data_dir):

    images = sorted(glob.glob(os.path.join(data_dir, f'*_eopatch-*.tif')))
    patches = list(set([int(os.path.basename(image).split('_')[4].split('-')[1]) for image in images]))

    return sorted(patches)


def main():
    data_dir = PARAMS.data_dir
    output_dir = PARAMS.output_dir
    match_band = PARAMS.match_band

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ref_date = datetime(2020, 7, 1)

    data = []

    for eopatch_idx in get_patch_indices(data_dir):

        print(f'eopatch-{eopatch_idx:04d}')

        image_paths = sorted(glob.glob(os.path.join(data_dir, f'*_eopatch-{eopatch_idx:04d}-{eopatch_idx:04d}.tif')))
        image_names = [os.path.basename(image_path) for image_path in image_paths]

        timestamps = [os.path.basename(image_name).split('_')[3] for image_name in image_names]
        dates = np.array([datetime.strptime(timestamp, '%Y-%m-%d-%H-%M-%S') for timestamp in timestamps])

        closest_date_id = np.argsort(abs(ref_date - dates))[0]

        image_names = image_names[closest_date_id + 1:] + image_names[:closest_date_id + 1]

        # Copy reference base image since no need to co-register
        if not os.path.exists(os.path.join(output_dir, image_names[-1])):
            shutil.copy(os.path.join(data_dir, image_names[-1]), os.path.join(output_dir, image_names[-1]))

        # Co-register with latest co-registered image (the reference base is assumed to be co-registered)
        for idx in range(len(image_names) - 1, 0, -1):
            if os.path.exists(os.path.join(output_dir, image_names[idx - 1])):
                continue
            fp = tempfile.NamedTemporaryFile(suffix='.tif')
            CR = COREG(os.path.join(output_dir, image_names[idx]),
                       os.path.join(data_dir, image_names[idx - 1]),
                       path_out=fp.name, fmt_out='GTIFF',
                       r_b4match=match_band, s_b4match=match_band, q=True, v=False)
            try:
                result = CR.calculate_spatial_shifts()
            except RuntimeError:
                result = 'fail'
            if result == 'success':
                _ = CR.correct_shifts()
                (x_min, y_min, x_max, y_max) = CR.shift.footprint_poly.bounds
                subprocess.check_call(['gdalwarp', '-te', str(x_min), str(y_min), str(x_max), str(y_max),
                                       fp.name,
                                       os.path.join(output_dir, image_names[idx - 1])],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                data.append([os.path.splitext(image_names[idx - 1])[0], CR.coreg_info])
            else:
                print(f'Failed to register {image_names[idx - 1]} to {image_names[idx]} using band [{match_band}]')
                shutil.copy(os.path.join(data_dir, image_names[idx - 1]),
                            os.path.join(output_dir, image_names[idx - 1]))
            fp.close()

    df = pd.DataFrame(data, columns=['Image', 'COREG Info'])
    df.to_pickle(os.path.join(output_dir, 'coreg_info.pkl'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Dataset directory (TIFF)", required=True)
    parser.add_argument("--output_dir", help="Output directory", required=True)
    parser.add_argument("--match_band", help="Band used for matching reference and target images", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()