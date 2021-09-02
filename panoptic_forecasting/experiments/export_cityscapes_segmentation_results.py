# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import glob
import os

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
import cv2
import cityscapesscripts.helpers.labels as cslabels


def convert_labels(seg):
    new_seg = torch.zeros_like(seg)
    for orig_cl in range(19):
        new_cl = cslabels.trainId2label[orig_cl].id
        new_seg[seg == orig_cl] = new_cl
    return new_seg

def convert_labels_to_trainid(seg):
    new_seg = torch.zeros_like(seg)
    for id, label in csscripts.id2label.items():
        new_seg[seg == id] = label.trainId
    return new_seg
    


def make_color_seg(seg):
    h, w = seg.shape
    result = np.zeros((h, w, 3))
    vals = np.unique(seg)
    for val in vals:
        label = cslabels.id2label[val]
        color = label.color
        result[seg == val] = color
    return result


def export_results(model, dataset, split, params):
    no_gpu = params.get('no_gpu', False)
    batch_size = params['training']['batch_size']
    collate_fn = params.get('collate_fn', None)
    num_workers = params['training'].get('num_data_workers', 0)
    working_dir = params['working_dir']
    no_convert = params.get('no_convert')
    convert2trainid = params.get('convert_to_trainid')
    print("NO CONVERT: ",no_convert)
    viz = params['viz']
    is_img = params['is_img']
    save_depth = params['save_depth']
    save_disp_as_png = params['save_disp_as_png']
    save_depth_as_png = params['save_depth_as_png']
    disp_factor = params['disp_factor']
    export_name = params['export_name']
    if export_name is not None:
        base_result_dir = os.path.join(working_dir, export_name, split)
    elif viz:
        base_result_dir = os.path.join(working_dir, 'exported_predictions_viz', split)
    else:
        base_result_dir = os.path.join(working_dir, 'exported_predictions', split)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn = collate_fn,
                             num_workers=num_workers, pin_memory=False)
    num_classes = params['data']['num_classes']
    for batch_ind, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        if not no_gpu:
            batch = train_utils.batch2gpu(batch)
        inputs = batch['inputs']
        labels = batch['labels']
        meta = batch['meta']
        with torch.no_grad():
            preds = model.predict(inputs, labels)
        pred_seg = preds['seg']
        for b_ind in range(len(pred_seg)):
            seg = pred_seg[b_ind]
            if not no_convert and not is_img:
                seg = convert_labels(seg)
            elif convert2trainid and not is_img:
                seg = convert_labels_to_trainid(seg)
            city = meta['city'][b_ind]
            seq = meta['seq'][b_ind]
            frame = meta['frame'][b_ind]
            target_frame = meta['target_frame'][b_ind]
            out_dir = os.path.join(base_result_dir, city)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if viz:
                out_path = os.path.join(out_dir, '%s_%s_%06d_gtFine_color.png'%(city, seq, target_frame))
                color_seg = make_color_seg(seg.cpu().numpy())
                im = Image.fromarray(color_seg.astype(np.uint8))
            elif is_img:
                out_path = os.path.join(out_dir, '%s_%s_%06d_leftImg8bit.png'%(city, seq, target_frame))
                im = Image.fromarray(seg.cpu().numpy().astype(np.uint8))
            else:
                out_path = os.path.join(out_dir, '%s_%s_%06d_gtFine_labelIds.png'%(city, seq, target_frame))
                im = Image.fromarray(seg.cpu().numpy().astype(np.uint8))
            im.save(out_path)
            if save_depth:
                if save_disp_as_png:
                    disps = preds['depth'][b_ind]
                    disps[disps >= 0] = (disp_factor / disps[disps >= 0]).clamp(0, 255)*256
                    disps[disps < 0] = 0
                    disps = disps.round().cpu().numpy().astype(np.uint16)
                    out_path = os.path.join(out_dir, '%s_%s_%06d_disps.png'%(city, seq, target_frame))
                    cv2.imwrite(out_path, disps)
                elif save_depth_as_png:
                    depths = preds['depth'][b_ind]
                    depths = (depths+1).clamp(0, 255)*256
                    depths = depths.round().cpu().numpy().astype(np.uint16)
                    out_path = os.path.join(out_dir, '%s_%s_%06d_depths.png'%(city, seq, target_frame))
                    cv2.imwrite(out_path, depths)
                else:
                    out_path = os.path.join(out_dir, '%s_%s_%06d_depths.npy'%(city, seq, target_frame))
                    np.save(out_path, preds['depth'][b_ind].cpu().numpy())

    if viz or is_img:
        return
    # for any missing files, create all zeros file
    cityscapes_dir = params['data'].get('cityscapes_dir')
    if cityscapes_dir is None:
        print("DID NOT RECEIVE CITYSCAPES DIR. SKIPPING.")
        return
    print("CHECKING FOR MISSING FILES...")
    gt_dir = os.path.join(cityscapes_dir, 'gtFine', dataset.split)
    count = 0
    city_options = params['data'].get('cities')
    for city in os.listdir(gt_dir):
        if city_options is not None and city not in city_options:
            print("SKIPPING CHECK FOR ",city)
            continue
        city_glob = os.path.join(gt_dir, city, '*_gtFine_labelIds.png')

        for city_path in glob.glob(city_glob):
            fname = os.path.basename(city_path)
            out_name = os.path.join(base_result_dir, city, fname)
            if not os.path.exists(out_name):
                count += 1
                try:
                    background_dir = dataset.background_dir
                    print("FOUND BACKGROUND DIR")
                    path = os.path.join(background_dir, city, fname)
                    arr_img = np.array(Image.open(path))
                    tensor_img = torch.from_numpy(arr_img).long()
                    tensor_img = convert_labels(tensor_img)
                    img = Image.fromarray(tensor_img.numpy().astype(np.uint8))
                except:
                    print("NO BACKGROUND FOUND")
                    if no_convert:
                        img = Image.fromarray(np.ones((1024, 2048), dtype=np.uint8)*255)
                    else:
                        img = Image.fromarray(np.zeros((1024, 2048), dtype=np.uint8))
                img.save(out_name)
    print("NUM MISSING: ",count)



if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    extra_args=[
        ['--viz', {'action':'store_true'}],
        ['--is_img', {'action':'store_true'}],
        ['--save_depth', {'action':'store_true'}],
        ['--save_depth_as_png', {'action':'store_true'}],
        ['--save_disp_as_png', {'action':'store_true'}],
        ['--disp_factor', {'type':float}],
        ['--export_name', {}],
        ['--no_convert', {'action':'store_true'}],
        ['--convert_to_trainid', {'action':'store_true'}]
    ]
    params = load_config(extra_args)
    misc.seed(params['seed'])
    data = build_dataset(params, test=True)
    model = build_model(params)
    model.eval()
    for split, dataset in data.items():
        export_results(model, dataset, split, params)
