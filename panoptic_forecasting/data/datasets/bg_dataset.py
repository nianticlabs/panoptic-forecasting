# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import glob

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cityscapesscripts.helpers.labels as csscripts
import h5py
from tqdm import tqdm

from panoptic_forecasting.data import data_utils
from panoptic_forecasting.data import transforms

MOVING_TRAIN_IDS = [label.trainId for label in csscripts.labels
                    if label.hasInstances and not label.ignoreInEval]

class BGDataset(Dataset):
    def __init__(self, split, params, test=False):
        self.test = test
        data_dir = params['data']['data_dir']
        self.data_inp_size = params['data'].get('data_inp_size', 3)
        if isinstance(data_dir, list):
            self.multiple_imgs = True
            self.data_dir = [os.path.join(d, split) for d in data_dir]
            full_data_dir = []
            for st in range(0, len(self.data_dir), self.data_inp_size):
                full_data_dir.append(self.data_dir[st:st+self.data_inp_size])
            self.data_dir = full_data_dir

        else:
            self.multiple_imgs = False
            self.data_dir = os.path.join(data_dir, split)
        cityscapes_dir = params['data']['cityscapes_dir']
        self.gt_dir = os.path.join(params['data']['gt_dir'], split)
        self.split = split
        self.load_depths = params['data'].get('load_depths')
        self.depth_h5_path = params['data'].get('depth_h5_path')
        self.depth_h5_path = self.depth_h5_path % split
        self.depth_h5 = None
        self.crop_size = params['data'].get('crop_size')
        self.scale_min = params['data'].get('scale_min')
        self.scale_max = params['data'].get('scale_max')
        self.use_depths = params['data'].get('use_depths')
        self.min_depth = params['data'].get('min_depth')
        self.max_depth = params['data'].get('max_depth')
        self.expand_test = params['data'].get('expand_test')
        self.depth_norm_params_file = params['data'].get('depth_norm_params_file')
        self.resize_w = params['data'].get('resize_w')
        self.resize_h = params['data'].get('resize_h')
        self.gap_len = params['data'].get('gap_len', [9,]) # default corresponds to medium term


        self.only_background = params['data'].get('only_background')
        if self.only_background:
            self.num_classes = params['data']['num_classes'] = 11
        else:
            self.num_classes = params['data']['num_classes'] = 19
        params['collate_fn'] = collate_fn

        self.data = []
        all_depths = []
        if params.get('continue_training') or self.test:
            compute_depth = False
        elif os.path.exists(self.depth_norm_params_file):
            depth_mean, depth_std = torch.load(self.depth_norm_params_file)
            compute_depth = False
        else:
            compute_depth = True
        print("COMPUTE DEPTH: ",compute_depth)
        for city in tqdm(os.listdir(self.gt_dir)):
            gt_city_dir = os.path.join(self.gt_dir, city)
            gt_glob = os.path.join(gt_city_dir, '*_labelTrainIds.png')
            for file_idx, gt_file in enumerate(glob.glob(gt_glob)):
                name = os.path.basename(gt_file)
                name_parts = name.split('_')
                city = name_parts[0]
                seq = name_parts[1]
                frame = int(name_parts[2])
                for data_dirs, gap_len in zip(self.data_dir, self.gap_len):
                    start_fr = (9 - gap_len)/3
                    frs = [19]
                    for fr in frs:
                        gt_fr = frame - 19 + fr
                        seg_name_parts = name_parts[:2] + ['%06d'%gt_fr, 'gtFine', 'labelIds.png']
                        if self.multiple_imgs:
                            data_file = []
                            for d in data_dirs:
                                file = os.path.join(d, city, '_'.join(seg_name_parts))
                                data_file.append(file)
                        else:
                            data_file = [os.path.join(data_dirs, city,
                                                     '_'.join(seg_name_parts))]
                            if not os.path.exists(data_file):
                                raise IOError('Could not find data file: ',data_file)
                        if fr == 19:
                            actual_gt = gt_file
                        else:
                            if not self.test:
                                # we currently only allow supervising with GT annotated frame
                                raise NotImplementedError()
                        start_fr 
                        self.data.append((actual_gt, data_file, city, seq, frame,
                                          fr, start_fr))

                if self.split != 'train' or self.test:
                    continue
                if self.use_depths and self.split == 'train' and compute_depth and file_idx%5 == 0:
                    if self.depth_h5 is None:
                        self.depth_h5 = h5py.File(self.depth_h5_path, 'r')
                    name = '%s/%s/%06d/%d'%(city, seq, frame, start_fr)
                    depths = self.depth_h5[name][:]
                    depths = self._clamp_depths(depths)
                    all_depths.append(depths[depths > 0])
                
        if self.split == 'train' and not self.test:
            if self.use_depths:
                self.depth_h5 = None
                if compute_depth:
                    all_depths = np.concatenate(all_depths)
                    while True:
                        try:

                            depth_mean = np.mean(all_depths)
                            depth_std = np.std(all_depths)
                            depth_mean = torch.FloatTensor([depth_mean])
                            depth_std = torch.FloatTensor([depth_std])
                            torch.save([depth_mean, depth_std],
                                       self.depth_norm_params_file)
                            break
                        except MemoryError:
                            all_depths = all_depths[:int(-all_depths.shape[0]/2)]
                if not params['continue_training']:
                    params['data']['depth_norm_params'] = [depth_mean, depth_std]
                    print("NORM PARAMS: ",depth_mean, depth_std)
            self.no_resize_crop = params['data'].get("no_resize_crop")
            if self.no_resize_crop:
                self.transforms = [transforms.RandomHorizontallyFlip()]
            else:
                self.transforms = [
                    transforms.RandomSizeAndCropMasks_Faster(self.crop_size,
                                                    False,
                                                    pre_size=None,
                                                    scale_min=self.scale_min,
                                                    scale_max=self.scale_max,
                                                    ignore_index=255,),
                    transforms.RandomHorizontallyFlip()
                ]
            if self.resize_h is not None:
                self.transforms.insert(0, transforms.Resize((self.resize_w, self.resize_h)))
        else:
            self.transforms = []
            if self.resize_h is not None:
                self.transforms.insert(0, transforms.Resize((self.resize_w, self.resize_h)))

    def __len__(self):
        return len(self.data)

    def _clamp_depths(self, depths):
        mask = depths > 0
        depths[mask & (depths > self.max_depth)] = self.max_depth
        depths[mask & (depths < self.min_depth)] = self.min_depth
        return depths

    def __getitem__(self, idx):
        (
            gt_file, data_file, city, seq, frame,
            fr, start_fr
        ) = self.data[idx]
        gt_fr = frame - 19 + fr

        gt_seg_img = Image.open(gt_file)

        data_seg_imgs = [Image.open(f) for f in data_file]

        if self.use_depths:
            if self.depth_h5 is None:
                self.depth_h5 = h5py.File(self.depth_h5_path, 'r')
            name = '%s/%s/%06d/%d'%(city, seq, frame, start_fr)
            arrs = [self.depth_h5[name][:]]
        else:
            arrs = []

        for tr_ind, transform in enumerate(self.transforms):
            data_seg_imgs, gt_seg_img, arrs = transform(data_seg_imgs, gt_seg_img, arrs)
        arr_ind = 0
        if self.use_depths:
            depth_arr = arrs[arr_ind]
            depth_arr = [depth_arr[:, :, idx] for idx in range(depth_arr.shape[2])]
            arr_ind += 1
        gt_seg = torch.from_numpy(
            np.array(gt_seg_img, dtype=np.int32)
        ).long()


        data_segs = [np.array(seg) for seg in data_seg_imgs]
        data_seg = torch.from_numpy(
            np.stack(data_segs)
        ).long()

        result = {
            'inputs':{
                'seg': data_seg,
            },
            'labels':{
                'seg': gt_seg,
            },
            'meta': {
                'city': city,
                'seq': seq,
                'frame': frame,
                'start_frame': start_fr,
                'target_frame': gt_fr,
            },
        }
        if self.use_depths:
            depths = torch.from_numpy(np.stack(depth_arr).astype(np.float32)).float()
            depths = depths / 256.0 - 1
            depth_masks = depths > 0
            depths[~depth_masks] = -1
            depths = self._clamp_depths(depths)
            result['inputs']['depth'] = depths
            result['inputs']['depth_mask'] = depth_masks

        return result


def collate_fn(batch):
    inputs = [entry['inputs'] for entry in batch]
    labels = [entry['labels'] for entry in batch]
    meta = [entry['meta'] for entry in batch]
    is_background = torch.utils.data.get_worker_info() is not None
    result_inp = {}
    for key in inputs[0]:
        out = None
        if is_background:
            tmp = inputs[0][key]
            numel = sum([entry[key].numel() for entry in inputs])
            storage = tmp.storage()._new_shared(numel)
            out = tmp.new(storage)
        result_inp[key] = torch.stack([entry[key] for entry in inputs], out=out)
    result_label = {}
    for key in labels[0]:
        out = None
        if is_background:
            tmp = labels[0][key]
            numel = sum([entry[key].numel() for entry in labels])
            storage = tmp.storage()._new_shared(numel)
            out = tmp.new(storage)
        result_label[key] = torch.stack([entry[key] for entry in labels], out=out)
    result_meta = {}
    for key in meta[0]:
        result_meta[key] = [entry[key] for entry in meta]
    return {'inputs': result_inp, 'labels': result_label, 'meta': result_meta}

def build_dataset(params, test=False):
    splits = params['data']['data_splits']
    datasets = {split:BGDataset(split, params, test=test) for split in splits}
    return datasets