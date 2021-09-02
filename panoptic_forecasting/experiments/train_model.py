# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from panoptic_forecasting.utils.config import load_config
from panoptic_forecasting.models import build_model
from panoptic_forecasting.data import build_dataset
import panoptic_forecasting.training.train as train
import panoptic_forecasting.training.train_utils as train_utils
import panoptic_forecasting.utils.misc as misc

import os

if __name__ == '__main__':
    params = load_config()

    misc.seed(params['seed'])
    misc.copy_config(params)

    datasets = build_dataset(params)
    model = build_model(params)
    with train_utils.build_writers(
            params['working_dir'], params['data']['data_splits']) as writers:
        train.train(model, datasets, params, writers)
