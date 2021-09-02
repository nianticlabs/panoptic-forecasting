# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from .datasets import bg_dataset
from .datasets import fg_instance_dataset
from .datasets import fg_scene_dataset
from .datasets import odom_dataset
from .datasets import pc_transform_dataset


def build_dataset(params, test=False):
    task = params['task']
    if task == 'fg':
        dataset_type = params['data']['dataset_type']
        if dataset_type == 'fg_instance':
            return fg_instance_dataset.build_dataset(params, test=test)
        elif dataset_type == 'fg_scene':
            return fg_scene_dataset.build_dataset(params, test=test)
        else:
            raise ValueError('dataset_type not recognized:', dataset_type)
    elif task == 'bg':
        return bg_dataset.build_dataset(params, test=test)
    elif task == 'odom':
        return odom_dataset.build_dataset(params, test=test)
    elif task == 'pc_transform':
        return pc_transform_dataset.build_dataset(params, test=test)
    else:
        raise ValueError('task not recognized: ',task)