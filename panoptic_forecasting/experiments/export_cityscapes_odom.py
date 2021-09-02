# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import h5py

from panoptic_forecasting.data import build_dataset
from panoptic_forecasting.models import build_model
import panoptic_forecasting.utils.misc as misc
import panoptic_forecasting.training.train_utils as train_utils
from panoptic_forecasting.utils.config import load_config

from tqdm import tqdm

def export_results(model, dataset, split, params):
    no_gpu = params.get('no_gpu', False)
    batch_size = params['training']['batch_size']
    collate_fn = params.get('collate_fn', None)
    num_workers = params['training'].get('num_data_workers', 0)
    working_dir = params['working_dir']
    no_convert = params.get('no_convert')
    print("NO CONVERT: ",no_convert)
    export_name = params['export_name']
    if export_name is not None:
        out_file = os.path.join(working_dir, '%s_%s.h5'%(export_name, split))
    else:
        out_file = os.path.join(working_dir, 'odometry_%s.h5'%split)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn = collate_fn,
                             num_workers=num_workers)
    with h5py.File(out_file, 'w') as fout:
        for batch_ind, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            if not no_gpu:
                batch = train_utils.batch2gpu(batch)
            inputs = batch['inputs']
            labels = batch['labels']
            meta = batch['meta']
            with torch.no_grad():
                preds = model.predict(inputs, labels)
            odom = preds['odometry'].cpu().numpy()
            for b_ind in range(len(odom)):
                odom_i = odom[b_ind]
                city = meta['city'][b_ind]
                seq = meta['seq'][b_ind]
                frame = meta['frame'][b_ind]
                start_frame = meta['start_frame'][b_ind]
                name = '%s/%s/%d/%d'%(city, seq, frame, start_frame)
                fout.create_dataset(name, data=odom_i)


if __name__ == '__main__':
    extra_args = [
        ['--export_name', {}],
    ]
    params = load_config(extra_args)
    misc.seed(params['seed'])
    data = build_dataset(params, test=True)
    model = build_model(params)
    model.eval()
    for split, dataset in data.items():
        export_results(model, dataset, split, params)
