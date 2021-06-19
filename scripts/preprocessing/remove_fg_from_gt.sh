# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

export CITYSCAPES_DATASET='data/cityscapes/'
python -m cityscapesscripts.preparation.createTrainIdLabelImgs

in_dir='data/cityscapes/gtFine/train/'
out_dir='data/cityscapes/gtFine_nofg/train'
python -u scripts/preprocessing/remove_fg_from_gt.py --input_dir $in_dir --output_dir $out_dir

in_dir='data/cityscapes/gtFine/val/'
out_dir='data/cityscapes/gtFine_nofg/val/'
python -u scripts/preprocessing/remove_fg_from_gt.py --input_dir $in_dir --output_dir $out_dir