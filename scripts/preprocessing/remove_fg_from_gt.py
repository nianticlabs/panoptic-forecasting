# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import argparse
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import cityscapesscripts.helpers.labels as csscripts

MOVING_TRAIN_IDS = [label.trainId for label in csscripts.labels
                    if label.hasInstances and not label.ignoreInEval]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')

    args = parser.parse_args()

    for city in tqdm(os.listdir(args.input_dir)):
        city_dir = os.path.join(args.input_dir, city)
        gt_glob = os.path.join(city_dir, '*_gtFine_labelTrainIds.png')
        for gt_file in glob.glob(gt_glob):
            fname = os.path.basename(gt_file)
            seg = np.array(Image.open(gt_file))
            for id in MOVING_TRAIN_IDS:
                seg[seg == id] = 255
            out_dir = os.path.join(args.output_dir, city)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)
            im = Image.fromarray(seg)
            im.save(out_path)

