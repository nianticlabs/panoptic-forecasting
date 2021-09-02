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
from PIL import Image
import cityscapesscripts.helpers.labels as csscripts
import h5py
from tqdm import tqdm
import cv2

from panoptic_forecasting.data import data_utils


class PCTransformDataset(Dataset):
    def __init__(self, split, params):
        self.data_dir = params['data']['data_dir']
        self.cityscapes_dir = params['data']['cityscapes_dir']
        self.no_moving_objects = params['data'].get('no_moving_objects')
        self.use_all_targets = params['data'].get('use_all_targets')
        self.expand_test = params['data'].get('expand_test')
        self.use_cascade_disps = params['data'].get('use_cascade_disps')
        self.use_mono = params['data'].get('use_mono_disps')
        self.use_orbslam_odom = params['data'].get('use_orbslam_odom')
        self.odom_pred_dir = params['data'].get('odom_pred_dir')
        self.cities = params['data'].get('cities')
        self.use_imgs = params['data'].get('use_imgs')
        self.monodepth_factor = params['data'].get('monodepth_factor', 5.405405405405405)
        if self.odom_pred_dir is not None:
            print("USING PREDICTED ODOM")
            odom_name = params['data'].get('odom_name', 'odometry')
            self.odom_pred_dir = os.path.join(self.odom_pred_dir,
                                              '%s_%s.h5'%(odom_name, split))
        self.cam_dir = os.path.join(self.cityscapes_dir, 'camera', split)
        self.timestamp_dir = os.path.join(self.cityscapes_dir, 'timestamp_sequence', split)
        self.odom_dir = os.path.join(self.cityscapes_dir, 'vehicle_sequence', split)
        self.check_output_dir = params['data'].get('check_output_dir')
        self.gap_len = params['data'].get('gap_len', 9) # default corresponds to medium term
        if self.use_cascade_disps:
            self.disparity_dir = params['data'].get('disparity_dir')
        elif self.use_mono:
            self.disparity_dir = os.path.join(params['data'].get('disparity_dir'), split)
        else:
            self.disparity_dir = os.path.join(
                self.cityscapes_dir, 'disparity_sequence', split,
            )
        if self.use_imgs:
            self.seg_dir = os.path.join(self.cityscapes_dir, 'leftImg8bit_sequence', split)
            if self.no_moving_objects:
                self.actual_seg_dir = os.path.join(params['data']['seg_dir'], split)
        else:
            self.seg_dir = os.path.join(
                params['data']['seg_dir'], split)
        self.split = split
        params['data']['num_classes'] = 19
        params['collate_fn'] = collate_fn
        if self.use_orbslam_odom:
            meta_path = os.path.join(self.data_dir, 'orbslam_odom_%s.pkl'%split)
        else:
            meta_path = os.path.join(self.data_dir, '%s_3d_info.pkl' % split)
        self.data = pd.read_pickle(meta_path)
        self.moving_object_inds = [id for id,label in csscripts.id2label.items()
                                   if label.hasInstances]
        def remove_moving(x):
            if x in self.moving_object_inds:
                return False
            else:
                return True
        self.remove_moving = np.vectorize(remove_moving)
        self.items = []
        if (self.split == 'train' and self.use_all_targets) or self.expand_test:
            targets = np.arange(6 + self.gap_len, 30)
            print("USING ALL TARGETS")
        else:
            targets = [19]
        base_input_inds = np.array([0, 3, 6])
        self.ego_transforms = {}
        self.ind_dict = {}
        for idx in tqdm(range(len(self.data))):
            datum = self.data.iloc[idx]
            city = datum['city']
            if self.cities is not None and city not in self.cities:
                continue
            seq = datum['seq']
            frame = datum['frame']
            for target in targets:
                input_inds = base_input_inds + target - (6+self.gap_len)
                if self.check_output_dir is not None:
                    fr = frame - 19 + target
                    test_file = os.path.join(self.check_output_dir, split,
                                             city, '%s_%s_%06d_gtFine_labelIds.png'%(city, seq, fr))
                    if os.path.exists(test_file):
                        continue
                self.items.append((idx, input_inds, target))
                self.ind_dict[(city, seq, frame)] = idx
            if self.odom_pred_dir is None:
                ego_transforms = []
                times = []
                odometries = []
                for fr in range(frame-19, frame+11):
                    time_path = os.path.join(self.timestamp_dir, city,
                                             '%s_%s_%06d_timestamp.txt'%(city, seq, fr))
                    with open(time_path, 'r') as fin:
                        time = float(fin.read())
                        times.append(time/1e9)
                    odom_path = os.path.join(self.odom_dir, city,
                                             '%s_%s_%06d_vehicle.json'%(city, seq, fr))
                    odom_dict = data_utils.read_json_file(odom_path)
                    speed = odom_dict.get('speed')
                    yaw_rate = odom_dict.get('yawRate')
                    if fr > frame - 19:
                        delta_t = times[-1] - times[-2]
                        T, dx, dy, dtheta = data_utils.get_vehicle_now_T_prev(
                            speed, yaw_rate, delta_t
                        )
                        ego_transforms.append(T)
                ego_transforms = np.stack(ego_transforms)
                self.ego_transforms[(city, seq, frame)] = ego_transforms
            else:
                times = []
                speeds = []
                yaw_rates = []
                odometries = []

                for fr in range(frame - 19, frame + 11):
                    time_path = os.path.join(self.timestamp_dir, city,
                                             '%s_%s_%06d_timestamp.txt' % (city, seq, fr))
                    with open(time_path, 'r') as fin:
                        time = float(fin.read())
                        times.append(time / 1e9)
                    if not self.use_orbslam_odom:
                        odom_path = os.path.join(self.odom_dir, city,
                                                '%s_%s_%06d_vehicle.json' % (city, seq, fr))
                        odom_dict = data_utils.read_json_file(odom_path)
                        speed = odom_dict.get('speed')
                        yaw_rate = odom_dict.get('yawRate')
                        speeds.append(speed)
                        yaw_rates.append(yaw_rate)
                if self.use_orbslam_odom:
                    speeds = list(datum['speed'])
                    yaw_rates = list(datum['yaw_rate'])
                for target in targets:
                    input_inds = base_input_inds + target - (6 + self.gap_len)
                    start_frame = input_inds[-1]
                    past_times = np.array(times[input_inds[0]:start_frame+1])
                    past_speeds = speeds[input_inds[0]+1:start_frame+1]
                    past_yaw_rates = yaw_rates[input_inds[0]+1:start_frame+1]
                    odom_name = '%s/%s/%d/%d'%(city, seq, frame, start_frame)
                    with h5py.File(self.odom_pred_dir, 'r') as fin:
                        odom_preds = fin[odom_name][:]
                    speed_preds = odom_preds[:self.gap_len, 0]
                    yaw_rate_preds = odom_preds[:self.gap_len, 1]
                    all_speeds = past_speeds + list(speed_preds)
                    all_yaw_rates = past_yaw_rates + list(yaw_rate_preds)
                    time_diffs = past_times[1:] - past_times[:-1]
                    all_time_diffs = list(time_diffs) + [np.mean(time_diffs) for
                                                     _ in range(len(speed_preds))]
                    ego_transforms = []
                    for o_idx in range(len(all_time_diffs)):
                        T, dx, dy, dtheta = data_utils.get_vehicle_now_T_prev(
                            all_speeds[o_idx], all_yaw_rates[o_idx],
                            all_time_diffs[o_idx]
                        )
                        ego_transforms.append(T)
                    cumulative_T = []
                    current_T = np.eye(4)
                    cumulative_T.append(current_T)
                    for fr in range(len(ego_transforms)-1, -1, -1):
                        egoT_t = ego_transforms[fr]
                        current_T = current_T @ egoT_t
                        cumulative_T.append(current_T)
                    cumulative_T.reverse()
                    if len(cumulative_T) != target - input_inds[0]+1:
                        print("EXPECTED: ", target-input_inds[0] + 1)
                        print("FOUND: ", len(cumulative_T))
                        sys.exit(0)
                    cumulative_T = np.stack(cumulative_T)[base_input_inds]

                    self.ego_transforms[(city, seq, frame, start_frame)] = cumulative_T
        print("NUM ITEMS: ", len(self.items))
        self.background_h5 = None

    def get_idx(self, city, seq, fr):
        idx = self.ind_dict[(city, seq, fr)]
        return idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        idx, input_inds, target = self.items[idx]
        datum = self.data.iloc[idx]
        city = datum['city']
        seq = datum['seq']
        frame = datum['frame']
        camera_path = os.path.join(self.cam_dir, city,
                                   '%s_%s_%06d_camera.json'%(city, seq, frame))
        camera = data_utils.read_json_file(camera_path)
        intrinsics = data_utils.cityscapes_camera2intrinsics(camera)
        extrinsics = data_utils.cityscapes_camera2extrinsics(camera)
        extrinsics = torch.from_numpy(extrinsics).float()
        baseline = camera.get('extrinsic').get('baseline')



        # we use fx for focal length 
        focal_length = intrinsics[0]


        K = torch.from_numpy(data_utils.build_intrinsics_mat(intrinsics)).float()

        if self.odom_pred_dir is None:
            ego_transforms = self.ego_transforms[(city, seq, frame)]
            cumulative_T = []
            current_T = np.eye(4)
            cumulative_T.append(current_T)
            for fr in range(target-1, -1, -1):
                egoT_t = ego_transforms[fr]
                current_T = current_T @ egoT_t
                cumulative_T.append(current_T)
            cumulative_T.reverse()
            cumulative_T = torch.from_numpy(
                np.stack(cumulative_T)[input_inds]
            ).float()
        else:
            cumulative_T = torch.from_numpy(
                self.ego_transforms[(city, seq, frame, input_inds[-1])]
            ).float()

        if self.use_imgs:
            base_seg_path = os.path.join(self.seg_dir, city,
                                         '%s_%s_%06d_leftImg8bit.png')
            if self.no_moving_objects:
                base_mask_path = os.path.join(self.actual_seg_dir, city,
                                        'pred_mask_%s_%s_%06d_leftImg8bit.png')
        else:
            base_seg_path = os.path.join(self.seg_dir, city,
                                        'pred_mask_%s_%s_%06d_leftImg8bit.png')
        if self.use_cascade_disps:
            base_disp_path = os.path.join(self.disparity_dir,
                                          '%s_%s_%06d_leftImg8bit.png')
        elif self.use_mono:
            base_disp_path = os.path.join(self.disparity_dir, city,
                                          '%s_%s_%06d_leftImg8bit_disp.npy')
        else:
            base_disp_path = os.path.join(self.disparity_dir, city,
                                          '%s_%s_%06d_disparity.png')
        inp_segs = []
        inp_depths = []
        inp_depth_masks = []
        
        for i,inp_ind in enumerate(input_inds):
            fr = frame - (19-inp_ind)
            seg_path = base_seg_path%(city, seq, fr)
            seg_img = Image.open(seg_path)
            seg_arr = np.array(seg_img)
            inp_segs.append(torch.from_numpy(seg_arr))

            disp_path = base_disp_path%(city, seq, fr)
            if self.use_mono:
                disps = np.load(disp_path)[0,0]
                disps = cv2.resize(disps, (2048, 1024), cv2.INTER_LINEAR)
                depths = self.monodepth_factor / disps
                depth_masks = np.ones_like(depths, dtype=bool)
            else:
                try:
                    depths, depth_masks = data_utils.load_depth(
                        disp_path, baseline, focal_length,
                        use_cascade=self.use_cascade_disps,
                    )
                except Exception as e:
                    print("OFFENDING PATH: ", disp_path)
                    raise e
            if self.no_moving_objects:
                if self.use_imgs:
                    mask_path = base_mask_path%(city, seq, fr)
                    mask_arr = np.array(Image.open(mask_path))
                    moving_mask = self.remove_moving(mask_arr)
                    depth_masks = depth_masks*moving_mask
                else:
                    moving_mask = self.remove_moving(seg_arr)
                    depth_masks = depth_masks * moving_mask
            
            inp_depths.append(torch.from_numpy(depths).float())
            inp_depth_masks.append(torch.from_numpy(depth_masks))

        inp_segs = torch.stack(inp_segs)
        inp_depths = torch.stack(inp_depths)
        inp_depth_masks = torch.stack(inp_depth_masks)
        result = {
            'inputs': {
                'seg': inp_segs,
                'depth': inp_depths,
                'depth_mask': inp_depth_masks,
                'intrinsics': K,
                'extrinsics': extrinsics,
                'target_T': cumulative_T,
            },
            'labels': {

            },
            'meta': {
                'city': city,
                'seq': seq,
                'frame': frame,
                'target_frame': frame - 19 + target,
            },
        }

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
    datasets = {split:PCTransformDataset(split, params) for split in splits}
    return datasets

