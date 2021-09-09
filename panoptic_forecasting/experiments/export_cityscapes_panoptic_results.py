# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import glob
import os
import json

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
import numpy as np

from panoptic_forecasting.data import build_dataset
from panoptic_forecasting.models import build_model
import panoptic_forecasting.utils.misc as misc
import panoptic_forecasting.training.train_utils as train_utils
from panoptic_forecasting.utils.config import load_config

from tqdm import tqdm
from PIL import Image
import cityscapesscripts.helpers.labels as cslabels


def convert_labels(seg):
    new_seg = np.zeros_like(seg)
    seg_vals = np.unique(seg)
    for seg_val in seg_vals:
        if seg_val == 255:
            new_seg_val = 0
        elif seg_val > 100:
            category_id = seg_val // 1000
            inst_id = seg_val % 1000
            new_cl = cslabels.trainId2label[category_id].id
            new_seg_val = new_cl * 1000 + inst_id
        else:
            new_seg_val = cslabels.trainId2label[seg_val].id
        new_seg[seg == seg_val] = new_seg_val
    return new_seg

def create_pan_img(seg):
    pan_img = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )
    for segmentId in np.unique(seg):
        mask = seg == segmentId
        color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
        pan_img[mask] = color
    pan_img = Image.fromarray(pan_img)
    return pan_img

def get_segments_info(seg):
    segments_info = []
    seg_vals = np.unique(seg)
    for seg_val in seg_vals:
        if seg_val == 0:
            continue
        elif seg_val > 100:
            category_id = int(seg_val / 1000)
        else:
            category_id = seg_val
        segments_info.append({
            "category_id": int(category_id),
            "id": int(seg_val),
        })
    return segments_info

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
        export_name = export_name + '_%s'%split
    else:
        export_name = 'exported_panoptics_%s'%split
    result_dir = os.path.join(working_dir, export_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    seg_dir = os.path.join(result_dir, export_name)
    os.makedirs(seg_dir, exist_ok=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn = collate_fn,
                             num_workers=num_workers, pin_memory=False)
    num_classes = params['data']['num_classes']
    final_annotations = []
    for batch_ind, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        if not no_gpu:
            batch = train_utils.batch2gpu(batch)
        inputs = batch['inputs']
        labels = batch['labels']
        meta = batch['meta']
        with torch.no_grad():
            preds = model.predict_panoptic(inputs, labels)
        pred_seg = preds['seg']
        for b_ind in range(len(pred_seg)):
            city = meta['city'][b_ind]
            seq = meta['seq'][b_ind]
            frame = meta['frame'][b_ind]
            target_frame = meta['target_frame'][b_ind]
            seg = pred_seg[b_ind]
            #print(city, seq, frame, torch.unique(seg))
            seg = seg.cpu().numpy()
            if not no_convert:
                seg = convert_labels(seg)
            #print("AFTER CONVERSION: ",np.unique(seg))
            segments_info = get_segments_info(seg)
            #print("SEGMENTS INFO: ",segments_info)
            new_annotations = {
                "file_name": '%s_%s_%06d_pred_panoptic.png'%(city, seq, target_frame),
                'image_id': '%s_%s_%06d'%(city, seq, target_frame),
                'segments_info': segments_info
            }
            final_annotations.append(new_annotations)
            pan_img = create_pan_img(seg)
            out_path = os.path.join(seg_dir, '%s_%s_%06d_pred_panoptic.png'%(city, seq, target_frame))
            pan_img.save(out_path)

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
        nobg=False
        for city_path in glob.glob(city_glob):
            fname = os.path.basename(city_path)
            tmp_info = fname.split('_')
            pred_name = '%s_%s_%s_pred_panoptic.png'%(tmp_info[0], tmp_info[1], tmp_info[2])
            out_name = os.path.join(seg_dir, pred_name)
            if not os.path.exists(out_name):
                count += 1
                try:
                    background_dir = dataset.background_dir
                    print("FOUND BACKGROUND DIR")
                    path = os.path.join(background_dir, city, fname)
                    arr_img = np.array(Image.open(path))
                    seg = convert_labels(arr_img)
                    segments_info = get_segments_info(seg)
                    new_annotations = {
                        "file_name": '%s_%s_%s_pred_panoptic.png' % (tmp_info[0], tmp_info[1], tmp_info[2]),
                        'image_id': '%s_%s_%s' % (tmp_info[0], tmp_info[1], tmp_info[2]),
                        'segments_info': segments_info,
                    }
                    final_annotations.append(new_annotations)
                    pan_img = create_pan_img(seg)
                except Exception as e:
                    nobg=True
                    pan_img = create_pan_img(np.zeros((1024, 2048), dtype=np.uint8))
                    new_annotations = {
                        "file_name": '%s_%s_%s_pred_panoptic.png' % (tmp_info[0], tmp_info[1], tmp_info[2]),
                        'image_id': '%s_%s_%s' % (tmp_info[0], tmp_info[1], tmp_info[2]),
                        'segments_info': [],
                    }
                    final_annotations.append(new_annotations)
                pan_img.save(out_name)
    if nobg:
        print("WARNING: NO BG DIR FOUND")
    print("NUM MISSING: ",count)
    print("NUM FINAL ANNOTATIONS: ",len(final_annotations))
    final_annotations = {"annotations": final_annotations}
    annotation_file = os.path.join(result_dir, '%s.json'%export_name)
    with open(annotation_file, 'w', encoding='utf-8') as f:
        json.dump(final_annotations, f, ensure_ascii=False, indent=4)




if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    extra_args=[
        ['--save_depth', {'action':'store_true'}],
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
