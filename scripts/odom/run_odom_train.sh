# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

GPU=0

config_file='./configs/odom/odom_train.yaml'
working_dir=./experiments/odom/
mkdir -p $working_dir
CUDA_VISIBLE_DEVICES=$GPU python -u panoptic_forecasting/experiments/train_model.py \
      --config_file $config_file \
      --working_dir $working_dir |& tee "${working_dir}results.txt"


#CUDA_VISIBLE_DEVICES=$GPU python -u panoptic_forecasting/experiments/train_model.py \
#      --continue_training \
#      --working_dir $working_dir |& tee "${working_dir}results_p2.txt"
