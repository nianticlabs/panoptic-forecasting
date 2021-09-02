# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

GPU=0

working_dir='./experiments/pretrained_fg/'
model_path='./pretrained_models/fg/fg_model.pt'



#############################################
# Mid Term
#############################################
config_file='./configs/fg/fg_val_mid.yaml'
export_name='exported_panoptics_midterm'
bg_dir='./experiments/pretrained_bg/exported_predictions_mid_trainids/'

CUDA_VISIBLE_DEVICES=$GPU python -u panoptic_forecasting/experiments/export_cityscapes_panoptic_results.py \
      --config_file $config_file \
      --load_model $model_path \
      --export_name ${export_name} \
      --extra_args data.background_dir $bg_dir \
      --working_dir $working_dir

python -m cityscapesscripts.evaluation.evalPanopticSemanticLabeling \
      --gt-json-file data/cityscapes/gtFine/cityscapes_panoptic_val.json \
      --gt-folder data/cityscapes/gtFine/cityscapes_panoptic_val/ \
      --prediction-json-file ${working_dir}${export_name}_val/${export_name}_val.json \
      --prediction-folder ${working_dir}${export_name}_val/${export_name}_val/ \
      --results_file ${working_dir}resultPanopticSemanticLabeling_midterm.json


#############################################
# Short Term
#############################################
config_file='./configs/fg/fg_val_short.yaml'
export_name='exported_panoptics_shortterm'
bg_dir='./experiments/pretrained_bg/exported_predictions_short_trainids/'

#CUDA_VISIBLE_DEVICES=$GPU python -u panoptic_forecasting/experiments/export_cityscapes_panoptic_results.py \
#      --config_file $config_file \
#      --load_model $model_path \
#      --export_name ${export_name} \
#      --working_dir $working_dir \
#      --extra_args data.background_dir $bg_dir
#
#python -m cityscapesscripts.evaluation.evalPanopticSemanticLabeling \
#      --gt-json-file data/cityscapes/gtFine/cityscapes_panoptic_val.json \
#      --gt-folder data/cityscapes/gtFine/cityscapes_panoptic_val/ \
#      --prediction-json-file ${working_dir}${export_name}_val/${export_name}_val.json \
#      --prediction-folder ${working_dir}${export_name}_val/${export_name}_val/ \
#      --results_file ${working_dir}resultPanopticSemanticLabeling_shortterm.json

