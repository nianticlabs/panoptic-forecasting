# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import glob
import os
from collections import defaultdict

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
import cityscapesscripts.helpers.labels as cslabels

from panoptic_forecasting.data import build_dataset
from panoptic_forecasting.models import build_model
import panoptic_forecasting.utils.misc as misc
import panoptic_forecasting.training.train_utils as train_utils
from panoptic_forecasting.utils.config import load_config


def convert_label(label):
    return cslabels.trainId2label[label].id




def export_results(model, dataset, split, params):
    no_gpu = params.get('no_gpu', False)
    batch_size = params['training']['batch_size']
    collate_fn = params.get('collate_fn', None)
    num_workers = params['training'].get('num_data_workers', 0)
    working_dir = params['working_dir']
    export_name = params['export_name']
    if export_name is not None:
        base_result_dir = os.path.join(working_dir, export_name, split)
    else:
        base_result_dir = os.path.join(working_dir, 'exported_instances', split)
    if not os.path.exists(base_result_dir):
        os.makedirs(base_result_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn = collate_fn,
                             num_workers=num_workers)
    entries = defaultdict(lambda: defaultdict(int))
    score_entries = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for batch_ind, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        if not no_gpu:
            batch = train_utils.batch2gpu(batch)
        inputs = batch['inputs']
        labels = batch['labels']
        meta = batch['meta']
        with torch.no_grad():
            preds = model.predict_instances(inputs, labels)
        pred_seg = preds['instances']
        classes = preds['instance_classes']
        scores = preds.get('instance_scores')
        for b_ind in range(len(pred_seg)):
            scene_segs = pred_seg[b_ind]
            city = meta['city'][b_ind]
            seq = meta['seq'][b_ind]
            frame = meta['frame'][b_ind]
            scene_classes = classes[b_ind]
            for scene_inst_ind in range(len(scene_segs)):
                seg = scene_segs[scene_inst_ind]*255

                cl = convert_label(scene_classes[scene_inst_ind].item())

                name = '%s_%s_%06d'%(city, seq, frame)
                inst_ind = entries[name][cl]
                entries[name][cl] += 1
                if scores is not None:
                    score = scores[b_ind][scene_inst_ind]
                else:
                    score = 1.0
                score_entries[name][cl][inst_ind] = score
                out_path = os.path.join(base_result_dir, '%s_%d_%d.png'%(name, cl, inst_ind))
                im = Image.fromarray(seg[0].cpu().numpy().astype(np.uint8))
                im.save(out_path)

    # Now: create txt files
    for name, cl_dict in entries.items():
        out_txt_path = os.path.join(base_result_dir, '%s.txt'%name)
        with open(out_txt_path, 'w') as fout:
            for cl, count in cl_dict.items():
                for i in range(count):
                    score = score_entries[name][cl][i]
                    tmp_name = '%s_%d_%d.png'%(name, cl, i)
                    fout.write('%s %d %f\n'%(tmp_name, cl, score))

    # for any missing files, create all zeros file
    cityscapes_dir = params['data'].get('cityscapes_dir')
    if cityscapes_dir is None:
        print("DID NOT RECEIVE CITYSCAPES DIR. SKIPPING.")
        return
    print("CHECKING FOR MISSING FILES...")
    gt_dir = os.path.join(cityscapes_dir, 'gtFine', dataset.split)
    count = 0
    for city in os.listdir(gt_dir):
        city_glob = os.path.join(gt_dir, city, '*_gtFine_labelIds.png')

        for city_path in glob.glob(city_glob):
            name = '_'.join(os.path.basename(city_path).split('_')[:3])
            if name not in entries:
                count += 1
                out_txt_path = os.path.join(base_result_dir, '%s.txt'%name)
                with open(out_txt_path, 'w') as _:
                    pass

    print("NUM MISSING: ",count)



if __name__ == '__main__':
    extra_args=[
        ['--export_name', {}],
        ['--no_convert', {'action':'store_true'}],
    ]
    params = load_config(extra_args)
    misc.seed(params['seed'])
    data = build_dataset(params, test=True)
    model = build_model(params)
    model.eval()
    for split, dataset in data.items():
        export_results(model, dataset, split, params)
