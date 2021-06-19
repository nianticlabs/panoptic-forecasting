# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import random
import yaml
import os

import torch
import numpy as np


def seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

def copy_config(params):
    working_dir = params['working_dir']
    config_path = os.path.join(working_dir, 'config.yaml')
    with open(config_path, 'w') as fout:
        yaml.dump(params, fout)