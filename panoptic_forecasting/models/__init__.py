# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

import torch

from .bg.bg_model import BGModel
from .fg.fg_model import FGModel
from .odom.odom_model import OdomModel
from .pc_transform.pc_transform_model import PCTransformModel

def build_model(params):
    task = params['task']
    print("Building model for task: ", task)
    if task == 'bg':
        model = BGModel(params)
    elif task == 'fg':
        model = FGModel(params)
    elif task == 'odom':
        model = OdomModel(params)
    elif task == 'pc_transform':
        model = PCTransformModel(params)
    else:
        raise ValueError('task not recognized: ',task)
    if not params['no_gpu']:
        if 'gpu' in params:
            device = torch.device('cuda')
            model.to(device)
        else:
            model.cuda()
    if params['load_best_model']:
        path = os.path.join(params['working_dir'], 'best_model')
        model.load(path)
    elif params['load_model']:
        print("LOADING MODEL FROM SPECIFIED PATH")
        model.load(params['load_model'])
    return model