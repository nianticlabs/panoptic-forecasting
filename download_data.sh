# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# Models
mkdir -p pretrained_models/fg
mkdir -p pretrained_models/bg
mkdir -p pretrained_models/odom
gsutil cp gs://niantic-lon-static/research/panoptic-forecasting/models/fg_model.pt pretrained_models/fg/fg_model.pt 
gsutil cp gs://niantic-lon-static/research/panoptic-forecasting/models/bg_model.pt pretrained_models/bg/bg_model.pt 
gsutil cp gs://niantic-lon-static/research/panoptic-forecasting/models/odom_model.pt pretrained_models/odom/odom_model.pt
gsutil cp gs://niantic-lon-static/research/panoptic-forecasting/models/predicted_odometry_train.h5 pretrained_models/odom/
gsutil cp gs://niantic-lon-static/research/panoptic-forecasting/models/predicted_odometry_val.h5 pretrained_models/odom/

# Data
mkdir -p data/
gsutil -m cp gs://niantic-lon-static/research/panoptic-forecasting/preprocessed-data/fg.tar.gz data/
tar -xzvf data/fg.tar.gz -C data/
gsutil -m cp gs://niantic-lon-static/research/panoptic-forecasting/preprocessed-data/bg.tar.gz data/
tar -xzvf data/bg.tar.gz -C data/