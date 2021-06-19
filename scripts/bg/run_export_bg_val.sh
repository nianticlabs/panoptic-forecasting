# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

GPU=0


working_dir='./experiments/pretrained_bg/'
model_path='./pretrained_models/bg/bg_model.pt'

################
# mid term
################
config_file='./configs/bg/bg_val_mid.yaml'
CUDA_VISIBLE_DEVICES=$GPU python -u panoptic_forecasting/experiments/export_cityscapes_segmentation_results.py \
      --config_file $config_file \
      --load_model $model_path \
      --no_convert \
      --export_name exported_predictions_mid_trainids \
      --working_dir $working_dir


#####################
# short term
######################
config_file='./configs/bg/bg_val_short.yaml'
CUDA_VISIBLE_DEVICES=$GPU python -u panoptic_forecasting/experiments/export_cityscapes_segmentation_results.py \
      --config_file $config_file \
      --load_model $model_path \
      --no_convert \
      --export_name exported_predictions_short_trainids \
      --working_dir $working_dir

