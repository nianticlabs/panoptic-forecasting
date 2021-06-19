# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

GPU=0


working_dir=./experiments/odom/

CUDA_VISIBLE_DEVICES=$GPU python -u panoptic_forecasting/experiments/export_cityscapes_odom.py \
      --load_best_model \
      --working_dir $working_dir
