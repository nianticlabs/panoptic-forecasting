# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
from torchvision import transforms
from PIL import Image

from panoptic_forecasting.data import data_utils

class OdomDataset(Dataset):
    def __init__(self, split, params, test=False):
        print("IS TEST: ",test)
        self.input_len = params['data'].get('input_len', 9)
        self.output_len = params['data'].get('output_len', 9)
        self.seq_len = self.input_len + self.output_len
        self.data_dir = params['data']['data_dir']
        params['collate_fn'] = collate_fn
        self.split = split
        self.test = test
        self.load_imgs = params['data'].get('load_imgs')
        self.cityscapes_dir = params['data'].get('cityscapes_dir')
        self.use_orbslam_odom = params['data'].get('use_orbslam_odom')

        if self.use_orbslam_odom:
            data_path = os.path.join(self.data_dir, 'orbslam_odom_%s.pkl'%split)
            self.data = pd.read_pickle(data_path)
            if self.split == 'train':
                all_speeds = np.stack(self.data['speed'])
                all_yaw_rates = np.stack(self.data['yaw_rate'])
                all_odometry = np.stack([all_speeds, all_yaw_rates], axis=-1)
                print("ALL ODOM SHAPE: ",all_odometry.shape)
                all_odometry = all_odometry.reshape(-1, 2)
                all_odom_means = torch.from_numpy(
                    all_odometry.mean(0)
                ).float()
                all_odom_stds = torch.from_numpy(
                    all_odometry.std(0)
                ).float()
                params['data']['odom_norm_params'] = (
                    all_odom_means, all_odom_stds
                )

        else:
            data_path = os.path.join(self.data_dir, '%s_3d_info.pkl'%split)
            self.data = pd.read_pickle(data_path)
            if self.split == 'train':
                all_odometry = np.stack(
                    self.data['odometry']
                ).reshape((-1, 5))[:, :2]
                all_odom_means = torch.from_numpy(
                    all_odometry.mean(0)
                ).float()
                all_odom_stds = torch.from_numpy(
                    all_odometry.std(0)
                ).float()
                params['data']['odom_norm_params'] = (
                    all_odom_means, all_odom_stds
                )
        self.inds = []
        for idx in tqdm(range(len(self.data))):
            datum = self.data.iloc[idx]
            inds = np.arange(18)
            if self.test:
                fr_range = range(30-self.input_len+1)
            else:
                fr_range = range(30-self.seq_len+1)
            for start_ind in fr_range:
                current_inds = (start_ind + inds).clip(max=29)
                self.inds.append((idx, start_ind, current_inds))
            self.inds.append((idx, -1, inds[:-1]))
            self.inds.append((idx, -2, inds[:-2]))
        if self.load_imgs:
            min_img_len = params['data'].get('min_img_len')
            self.transforms = transforms.Compose([
                transforms.Resize(min_img_len),
                transforms.ToTensor(),
            ])


    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        idx, start_ind, current_inds = self.inds[idx]
        datum = self.data.iloc[idx]
        if self.use_orbslam_odom:
            speeds = datum['speed']
            yaw_rate = datum['yaw_rate']
            odom = torch.from_numpy(np.stack([speeds, yaw_rate], axis=-1))[current_inds].float()
        else:
            odom = torch.from_numpy(datum['odometry'][current_inds, :2]).float()
        city = datum['city']
        seq = datum['seq']
        frame = datum['frame']
        if start_ind < 0:
            inp_odom = torch.cat([
                odom[0:1].repeat(-start_ind, 1),
                odom[:self.input_len + start_ind],
            ], dim=0)
            out_odom = odom[-self.output_len:]
            start_frame = current_inds[self.input_len-1+start_ind]
        else:
            inp_odom = odom[:self.input_len]
            out_odom = odom[self.input_len:]
            start_frame = current_inds[self.input_len-1]
        result = {
            'inputs': {
                'odometry': inp_odom,
            },
            'labels': {
                'odometry': out_odom,
            },
            'meta': {
                'city': city,
                'seq': seq,
                'frame': frame,
                'start_frame': start_frame,
            }
        }
        if self.load_imgs:
            base_img_path = os.path.join(self.cityscapes_dir, 'leftImg8bit_sequence',
                                         self.split, city,
                                         '%s_%s' % (city, seq) + '_%06d_leftImg8bit.png')
            imgs = []
            img_inds = current_inds[:self.input_len]
            if start_ind < 0:
                img_inds = current_inds[:self.input_len + start_ind]
            for ind in img_inds:
                fr = frame - 19 + ind
                img_path = base_img_path % fr
                img = Image.open(img_path)
                imgs.append(self.transforms(img))
            if start_ind < 0:
                for _ in range(-start_ind):
                    imgs.insert(0, imgs[0])

            imgs = torch.stack(imgs)
            result['inputs']['imgs'] = imgs
        return result


def collate_fn(batch):
    inputs = [entry['inputs'] for entry in batch]
    labels = [entry['labels'] for entry in batch]
    meta = [entry['meta'] for entry in batch]
    result_inp = {}
    for key in inputs[0]:
        result_inp[key] = torch.stack([entry[key] for entry in inputs])
    result_label = {}
    for key in labels[0]:
        result_label[key] = torch.stack([entry[key] for entry in labels])
    result_meta = {}
    for key in meta[0]:
        result_meta[key] = [entry[key] for entry in meta]
    return {'inputs': result_inp, 'labels': result_label, 'meta': result_meta}


def build_dataset(params, test=False):
    splits = params['data']['data_splits']
    datasets = {split:OdomDataset(split, params, test) for split in splits}
    return datasets
