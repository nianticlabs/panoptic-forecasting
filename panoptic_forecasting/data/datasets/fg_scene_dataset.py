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
from PIL import Image

from panoptic_forecasting.data import data_utils


class FGSceneDataset(Dataset):

    def __init__(self, split, params, test=False):
        self.data_dir = params['data']['data_dir']
        self.depth_dir = params['data']['depth_dir']
        self.use_3d_info = params['data'].get('use_3d_info')
        self.info_3d_dir = params['data'].get('info_3d_dir', self.data_dir)
        self.odom_pred_dir = params['data'].get('odom_pred_dir')
        self.use_cascade_depths = params['data'].get('use_cascade_depths')
        self.use_monodepth = params['data'].get('use_monodepth')
        if self.odom_pred_dir is not None:
            print("USING PREDICTED ODOM")
            odom_name = params['data'].get('odom_name', 'predicted_odometry')
            self.odom_pred_dir = os.path.join(self.odom_pred_dir,
                                              '%s_%s.h5'%(odom_name,split))
        self.no_feats = params['data'].get('no_feats')
        self.filter_car_gap = params['data'].get('filter_car_gap')
        self.filter_car_gap_borderdist = params['data'].get('filter_car_gap_borderdist',
                                                            self.filter_car_gap)
        self.max_depth = params['data'].get('max_depth')
        self.expand_train = params['data'].get('expand_train')
        self.expand_test = params['data'].get('expand_test')
        self.background_dir = params['data'].get('background_dir')
        self.input_len = params['data'].get('input_len', 3)
        self.cityscapes_dir = params['data'].get('cityscapes_dir')
        self.require_most_recent = params['data'].get('require_most_recent')
        self.output_ind = params['data'].get('output_ind')
        self.input_len = params['data'].get('input_len', 3)
        self.add_car_offscreen_loc = params['data'].get('add_car_offscreen_loc')
        if self.input_len != 3:
            raise NotImplementedError()
        if self.background_dir is not None:
            self.background_dir = os.path.join(self.background_dir,
                                               split)
        self.split = split
        self.test = test
        params['data']['num_classes'] = 19
        params['collate_fn'] = collate_fn
        meta_path = os.path.join(self.data_dir, '%s_seq_meta.pkl' % split)
        self.data = pd.read_pickle(meta_path)
        if self.use_cascade_depths:
            depth_path = os.path.join(self.depth_dir, '%s_cascadedepth_seq_info.pkl'%split) #TODO: CHECK PATH
        elif self.use_monodepth:
            depth_path = os.path.join(self.depth_dir, '%s_monodepth_seq_info.pkl'%split)
        else:
            depth_path = os.path.join(self.depth_dir, '%s_depth_seq_info.pkl'%split) #TODO: CHECK PATH
        self.depth_data = pd.read_pickle(depth_path)
        self.feats_dir = params['data'].get('feats_dir')
        self.use_condensed_feats = params['data'].get('use_condensed_feats')
        if self.use_condensed_feats:
            print("USING CONDENSED FEATS")
            self.feats_path = os.path.join(self.feats_dir, '%s_condensed_feats.h5'%split)
            feats_meta_path = os.path.join(self.feats_dir, '%s_seq_condensed_feat_info.pkl'%split)
            self.feats_meta = pd.read_pickle(feats_meta_path)
        else:
            self.feats_path = os.path.join(self.feats_dir, '%s_feats.h5' % split)
            if not self.no_feats and not os.path.exists(self.feats_path):
                self.feats_path = os.path.join(self.feats_dir, split, 'feats.h5')
        self.img_size = [2048, 1024]
        self.expanded_img_size = torch.FloatTensor(self.img_size)
        self.expanded_img_size = torch.cat([self.expanded_img_size,
                                            self.expanded_img_size])
        params['data']['img_size'] = torch.FloatTensor(self.img_size)
        self.traj_normalizer = torch.FloatTensor(self.img_size).repeat(4).unsqueeze(0)
        self.use_ulbr = params.get('use_bbox_ulbr')

        all_bboxes = np.concatenate(self.data['bboxes'].values)
        all_depths = np.concatenate(self.depth_data['depth'].values)
        if not self.use_ulbr:
            all_bboxes = data_utils.convert_bbox_ulbr_cwh(all_bboxes)
        all_feat_masks = np.concatenate(self.data['feat_mask'].values)
        all_depth_masks = (all_depths != -1)&(all_depths != 1000000)
        if self.max_depth is not None:
            all_depth_masks = (all_depth_masks)*(all_depths <= self.max_depth)
        self.instance_inds = []
        self.ind_dict = {}
        
        self.seq_len = 3
        self.output_len = 3
        inds = np.array([0, 3, 6, 9, 12, 15]).astype(int)
        if self.expand_train:
            start_inds = range(12)
        else:
            start_inds = [4, 7, 10]
        all_locs = []
        all_loc_masks = []
        total_depths = []
        total_depth_masks = []
        for start_ind in start_inds:
            all_locs.append(all_bboxes[:, inds + start_ind])
            all_loc_masks.append(all_feat_masks[:, inds + start_ind])
            total_depths.append(all_depths[:, inds + start_ind])
            total_depth_masks.append(all_depth_masks[:, inds + start_ind])
        all_locs = np.concatenate(all_locs)
        all_loc_masks = np.concatenate(all_loc_masks)
        total_depths = np.concatenate(total_depths)
        total_depth_masks = np.concatenate(total_depth_masks)

        if self.use_3d_info:
            self.use_orbslam_odom = params['data'].get('use_orbslam_odom')
            if self.use_orbslam_odom:
                data_path = os.path.join(self.info_3d_dir, 'orbslam_odom_%s.pkl'%split)
            else:
                data_path = os.path.join(self.info_3d_dir, '%s_3d_info.pkl'%split)
            self.data3d = pd.read_pickle(data_path)
        params['data']['odom_size'] = 5

        if self.split == 'train' and not self.test:
            final_locs = all_locs.reshape((-1, 4))[all_loc_masks.reshape((-1,))]
            mean_loc, std_loc = final_locs.mean(0), final_locs.std(0)
            all_vel_masks = all_loc_masks[:, 1:] * all_loc_masks[:, :-1]
            all_vels = all_locs[:, 1:] - all_locs[:, :-1]
            final_vels = all_vels.reshape((-1, 4))[all_vel_masks.reshape((-1,))]
            mean_vel, std_vel = final_vels.mean(0), final_vels.std(0)
            all_means = torch.from_numpy(
                np.concatenate([mean_loc, mean_vel])
            ).float()
            all_stds = torch.from_numpy(
                np.concatenate([std_loc, std_vel])
            ).float()
            params['data']['norm_params'] = (all_means, all_stds)

            final_depths = total_depths.reshape((-1,))[total_depth_masks.reshape((-1,))]
            mean_depth, std_depth = final_depths.mean(0), final_depths.std(0)
            all_vel_depth_masks = total_depth_masks[:, 1:] * total_depth_masks[:, :-1]
            all_depth_vels = total_depths[:, 1:] - total_depths[:, :-1]
            final_depth_vels = all_depth_vels.reshape((-1,))[all_vel_depth_masks.reshape((-1,))]
            mean_depth_vel, std_depth_vel = final_depth_vels.mean(), final_depth_vels.std()
            all_depth_means = torch.from_numpy(
                np.array([mean_depth, mean_depth_vel])
            ).float()
            all_depth_stds = torch.from_numpy(
                np.array([std_depth, std_depth_vel])
            ).float()
            params['data']['depth_norm_params'] = (all_depth_means, all_depth_stds)
            if self.use_3d_info:
                params['data']['odom_size'] = 5
                if self.use_orbslam_odom:
                    all_odometry = np.stack([
                        np.stack(self.data3d['speed']),
                        np.stack(self.data3d['yaw_rate']),
                        np.stack(self.data3d['dx']),
                        np.stack(self.data3d['dy']),
                        np.stack(self.data3d['dtheta']),
                    ], axis=-1)
                    all_odometry = all_odometry.reshape((-1, 5))
                else:
                    all_odometry = np.stack(
                        self.data3d['odometry']
                    ).reshape((-1, 5))
                all_odom_means = torch.from_numpy(
                    all_odometry.mean(0)).float()
                all_odom_stds = torch.from_numpy(
                    all_odometry.std(0)).float()

                params['data']['odom_norm_params'] = (
                    all_odom_means, all_odom_stds
                )
        for idx in tqdm(range(len(self.data))):
            datum = self.data.iloc[idx]
            feat_mask = np.array(datum['feat_mask'])
            city = datum['city']
            seq = datum['seq']
            frame = datum['frame']
            
            base_inds = np.arange(0, 3*(self.input_len + self.output_len), 3)
            if (self.split == 'train' and self.expand_train) or (test and self.expand_test):
                start_inds = range(30 - 3*(self.input_len + self.output_len-1))
            elif self.split == 'train':
                start_inds = [4, 7, 10]
            else:
                start_inds = [19 - 3*(self.input_len + self.output_len-1)]
            city = datum['city']
            seq = datum['seq']
            frame = datum['frame']
            feat_mask = np.array(datum['feat_mask'])
            self.ind_dict[(city, seq, frame)] = len(self.instance_inds)
            if self.split == 'train' or (test and self.expand_test):
                for start_ind in start_inds:
                    current_inds = start_ind + base_inds
                    current_feat_mask = feat_mask[:, current_inds]
                    current_feat_mask = current_feat_mask[:, :self.input_len]
                    if self.require_most_recent:
                        current_feat_mask = current_feat_mask[:, -1]
                    if np.any(current_feat_mask):
                        self.instance_inds.append((idx, 2, start_ind + base_inds, start_ind+base_inds))
            else:
                inds = np.array([4, 7, 10, 13, 16, 19]).astype(int)
                if self.output_ind == 0:
                    self.instance_inds.append((idx, 2, inds + 6, inds + 6))
                else:
                    self.instance_inds.append((idx, 0, inds, inds))
        print("TOTAL NUM INSTANCES: ",len(self.instance_inds))
        

    def __len__(self):
        return len(self.instance_inds)

    def _class2onehot(self, classes):
        result = torch.zeros(len(classes), 8)
        result[range(len(classes)), classes] = 1
        return result


    def _add_car_offscreen_loc(self, classes, bboxes, bbox_mask):
        bboxes = data_utils.convert_bbox_cwh_ulbr(bboxes.clone())
        bbox_mask = bbox_mask.clone()
        out_range = range(1, self.input_len + self.output_len)
        for inst_ind in range(len(bboxes)):
            if classes[inst_ind].item() != 13:
                continue
            completed = False
            for out_t in out_range:
                if completed:
                    break
                if not bbox_mask[inst_ind, out_t] and bbox_mask[inst_ind, out_t-1]:
                    if out_t < self.input_len - self.output_len-1 and np.any(bbox_mask[inst_ind, out_t+1:]):
                        continue
                    current_bbox = bboxes[inst_ind, out_t - 1].numpy()
                    x0, y0, x1, y1 = current_bbox
                    cx = (current_bbox[2] + current_bbox[0]) / 2
                    cy = (current_bbox[3] + current_bbox[1]) / 2
                    w = current_bbox[2] - current_bbox[0]
                    h = current_bbox[3] - current_bbox[1]
                    if current_bbox[0] < 200:
                        if out_t > 1 and bbox_mask[inst_ind, out_t - 2]:
                            o_bbox = bboxes[inst_ind, out_t - 2].numpy()
                            vx = x1 - o_bbox[2]
                            vy0 = y0 - o_bbox[1]
                            vy1 = y1 - o_bbox[3]
                            if vx > 0:
                                break
                            for tmp_t in range(out_t, self.input_len+self.output_len):
                                nx0 = x0 + vx
                                nx1 = x1 + vx
                                ny0 = y0 + vy0
                                ny1 = y1 + vy1
                                nx0 = max(nx0, -20)
                                nx1 = max(nx1, -10)
                                ny0 = min(ny0, self.img_size[1]+10)
                                ny1 = min(ny1, self.img_size[1]+20)
                                bboxes[inst_ind, tmp_t] = torch.FloatTensor([nx0, ny0, nx1, ny1])
                                bbox_mask[inst_ind, tmp_t] = True
                                x0 = nx0
                                x1 = nx1
                                y0 = ny0
                                y1 = ny1
                            completed = True

                    elif current_bbox[2] > self.img_size[0] - 200:
                        if out_t > 1 and bbox_mask[inst_ind, out_t - 2]:
                            o_bbox = bboxes[inst_ind, out_t - 2].numpy()
                            vx = x0 - o_bbox[0]
                            vy0 = y0 - o_bbox[1]
                            vy1 = y1 - o_bbox[3]
                            if vx < 0:
                                break
                            for tmp_t in range(out_t, self.input_len+self.output_len):
                                nx0 = x0 + vx
                                nx1 = x1 + vx
                                ny0 = y0 + vy0
                                ny1 = y1 + vy1
                                nx0 = min(nx0, self.img_size[0]+10)
                                nx1 = min(nx1, self.img_size[0]+10)
                                ny0 = min(ny0, self.img_size[1]+10)
                                ny1 = min(ny1, self.img_size[1]+20)
                                bboxes[inst_ind, tmp_t] = torch.FloatTensor([nx0, ny0, nx1, ny1])
                                bbox_mask[inst_ind, tmp_t] = True
                                x0 = nx0
                                x1 = nx1
                                y0 = ny0
                                y1 = ny1
                            completed = True
        bboxes = data_utils.convert_bbox_ulbr_cwh(bboxes)
        return bboxes, bbox_mask

    def get_idx(self, city, seq, fr):
        idx = self.ind_dict[(city, seq, fr)]
        return idx

    def _filter_car_gap(self, classes, bboxes, bbox_mask, feat_mask):
        bboxes = bboxes.clone()
        bbox_mask = bbox_mask.clone()
        feat_mask = feat_mask.clone()
        for inst_ind in range(len(bboxes)):
            if classes[inst_ind].item() != 13:
                continue
            past_loc = None
            found_x0 = False
            found_x1 = False
            zero_rest = False
            skip_this = False
            for tmp_idx in range(self.input_len + self.output_len):
                if not zero_rest:
                    if not bbox_mask[inst_ind, tmp_idx].item():
                        continue
                    x0, y0, x1, y1 = bboxes[inst_ind, tmp_idx]
                    if x0 < self.filter_car_gap:
                        found_x0 = True
                    if x1 > self.img_size[0] - self.filter_car_gap:
                        found_x1 = True
                    if found_x0:
                        if past_loc is None:
                            past_loc = x1
                        elif x1 > past_loc + self.filter_car_gap:
                            zero_rest = True
                        past_loc = x1
                    if found_x1:
                        if past_loc is None:
                            past_loc = x0
                        elif x0 < past_loc - self.filter_car_gap:
                            zero_rest = True
                        past_loc = x0
                if zero_rest:
                    if skip_this:
                        skip_this = False
                    else:
                        bbox_mask[inst_ind, tmp_idx] = 0
                        feat_mask[inst_ind, tmp_idx] = 0
                        bboxes[inst_ind, tmp_idx] = 0
        return bboxes, bbox_mask, feat_mask

    def __getitem__(self, idx):
        idx, start_fr, feat_inds, bbox_inds = self.instance_inds[idx]
        datum = self.data.iloc[idx]
        depth_datum = self.depth_data.iloc[idx]
        city = datum['city']
        seq = datum['seq']
        frame = datum['frame']

        fr_inds = feat_inds
        feat_masks = datum['feat_mask'][:, fr_inds]
        if self.use_condensed_feats:
            feat_datum = self.feats_meta.iloc[idx]
            feat_inds = feat_datum['feat_ind'][:, fr_inds]
        else:
            feat_inds = datum['feat_ind'][:, fr_inds]
        if self.require_most_recent:
            has_gt = feat_masks[:, self.input_len-1]
        else:
            has_gt = feat_masks[:, :self.input_len].sum(1) > 0
        feat_masks = np_feat_masks = feat_masks[has_gt]
        feat_inds = feat_inds[has_gt]
        feat_masks = torch.from_numpy(feat_masks)
        track_ids = datum['track_id'][has_gt]
        num_instances = track_ids.shape[0]
        img_size = torch.FloatTensor(self.img_size).repeat(num_instances, 1)

        bbox_mask = np.array(datum['feat_mask'])[has_gt][:, bbox_inds]
        bbox_mask = torch.from_numpy(bbox_mask)
        if self.output_ind is not None:
            bbox_output_inds = torch.LongTensor([self.output_ind for _ in range(num_instances)])
            output_inds = torch.LongTensor([self.output_ind for _ in range(num_instances)])
            target_frame = frame - 19 + fr_inds[self.input_len:][self.output_ind]
        else:
            bbox_output_inds = torch.LongTensor([self.seq_len-1 for _ in range(num_instances)])
            output_inds = torch.LongTensor([self.seq_len-1 for _ in range(num_instances)])
            target_frame = frame - 19 + fr_inds[self.input_len:][-1]

        classes = datum['class'][has_gt]
        bboxes = torch.from_numpy(datum['bboxes'][has_gt][:, bbox_inds]).float()

        if not self.use_ulbr:
            bboxes = data_utils.convert_bbox_ulbr_cwh(bboxes)
        if self.filter_car_gap is not None:
            bboxes, bbox_mask, feat_masks = self._filter_car_gap(classes, bboxes, bbox_mask, feat_masks)
        if self.add_car_offscreen_loc and not self.test:
            bboxes, bbox_mask = self._add_car_offscreen_loc(classes, bboxes, bbox_mask)
        

        bbox_vel = torch.cat([torch.zeros(num_instances, 1, 4),
                                bboxes[:, 1:] - bboxes[:, :-1]], dim=1)
        bbox_vel[:, 1:] *= (bbox_mask[:, :-1]*bbox_mask[:, 1:]).unsqueeze(-1)
        bbox_vel_mask = torch.cat([
            torch.zeros(len(bboxes), 1, dtype=bool),
            bbox_mask[:, 1:]*bbox_mask[:, :-1],
        ], dim=1)
        trajectories = torch.cat([bboxes, bbox_vel], dim=-1)
        depths = torch.from_numpy(depth_datum['depth'][has_gt][:, bbox_inds]).float().unsqueeze(-1)
        depth_masks = (depths != -1)&(depths != 1000000)
        if self.max_depth is not None:
            depth_masks = (depth_masks)*(depths <= self.max_depth)
        depth_vel = torch.cat([torch.zeros(num_instances, 1, 1),
                               depths[:, 1:] - depths[:, :-1]], dim=1)
        depth_vel[:, 1:] *= (depth_masks[:, :-1]*depth_masks[:, 1:])
        depths = torch.cat([depths, depth_vel], dim=-1)


        all_feats = []
        inp_classes = torch.LongTensor(classes) - 11
        one_hot_classes = self._class2onehot(inp_classes)
        result = {
            'inputs': {
                'feat_masks': feat_masks,
                'bbox_masks': bbox_mask,
                'bbox_vel_masks': bbox_vel_mask,
                'trajectories': trajectories[:, :self.seq_len],
                'depths': depths[:, :self.seq_len],
                'depth_masks': depth_masks[:, :self.seq_len],
                'classes': inp_classes,
                'one_hot_classes': one_hot_classes,
                'final_bboxes': torch.FloatTensor(bboxes[:, -1]),
                'img_size': img_size,
            },
            'labels': {
                'output_inds': output_inds,
                'bbox_output_inds': bbox_output_inds,
                'trajectories': trajectories[:, self.seq_len:],
                'depths': depths[:, self.seq_len:],
                'depth_masks': depth_masks[:, self.seq_len:],
            },
            'meta': {
                'city': city,
                'seq': seq,
                'frame': frame,
                'track_id': track_ids,
                'fr_inds': fr_inds,
                'target_frame': target_frame,
            }
        }
        if not self.no_feats:
            if num_instances > 0:
                with h5py.File(self.feats_path, 'r') as feats_in:
                    name = '%s/%s/%d'%(city, seq, frame)
                    dset = feats_in[name]
                    for tr_feat_inds, tr_feat_masks in zip(feat_inds, np_feat_masks):
                        feats = torch.zeros(len(tr_feat_inds), 256, 14, 14)
                        valid = tr_feat_inds != -1
                        tmp_feats = torch.from_numpy(dset[list(tr_feat_inds[valid])]).float()
                        feats[valid] = tmp_feats
                        all_feats.append(feats)
                all_feats = torch.stack(all_feats)
            else:
                all_feats = torch.zeros(0, 2*self.seq_len, 256, 14, 14)
            result['inputs']['feats'] = all_feats[:, :self.seq_len]
            result['labels']['feats'] = all_feats[:, self.seq_len:]
        if self.use_3d_info:
            try:
                datum3d = self.data3d[(self.data3d['city']==city) &
                                      (self.data3d['seq'] == seq) &
                                      (self.data3d['frame'] == frame)].iloc[0]
            except Exception as e:
                print("BAD POINT: ",city, seq, frame)
                raise e

            if self.odom_pred_dir is not None:
                if self.use_orbslam_odom:
                    inp_odom = np.stack([
                        datum3d['speed'],
                        datum3d['yaw_rate'],
                        datum3d['dx'],
                        datum3d['dy'],
                        datum3d['dtheta']
                    ], axis=-1)[bbox_inds[:self.input_len]]
                else:
                    inp_odom = datum3d['odometry'][bbox_inds[:self.input_len]]
                odom_name = '%s/%s/%d/%d'%(city, seq, frame, bbox_inds[self.input_len-1])
                inp_times = datum3d['times'][bbox_inds[0]:bbox_inds[self.input_len-1]+1]
                avg_delta_t = np.mean(inp_times[1:] - inp_times[:-1])
                with h5py.File(self.odom_pred_dir, 'r') as odom_in:
                    odom_preds = odom_in[odom_name][:]
                final_odom = []
                for odom_idx in range(len(odom_preds)):
                    speed, yaw_rate = odom_preds[odom_idx]
                    _, dx, dy, dtheta = data_utils.get_vehicle_now_T_prev(speed, yaw_rate, avg_delta_t)
                    final_odom.append(np.array([speed, yaw_rate, dx, dy, dtheta]))
                final_odom = np.stack(final_odom)
                final_odom = final_odom[[2,5,8]]
                odometry = torch.from_numpy(
                    np.concatenate([
                        inp_odom,
                        final_odom,
                    ])
                ).float()
            else:
                if self.use_orbslam_odom:
                    raise NotImplementedError()
                odometry = datum3d['odometry'][bbox_inds]
                odometry = torch.from_numpy(odometry).float()
            odometry = odometry.unsqueeze(0).expand(num_instances, -1, -1)
            result['inputs']['odometry'] = odometry
        if self.background_dir is not None:
            background_path = os.path.join(self.background_dir, city,
                                           '%s_%s_%06d_gtFine_labelIds.png'%(
                                               city, seq, target_frame,
                                           ))
            background = Image.open(background_path)
            background = torch.from_numpy(
                np.array(background, dtype=np.int32)
            ).long()
            result['inputs']['background'] = background
        return result


def collate_fn(batch):
    inputs = [entry['inputs'] for entry in batch]
    labels = [entry['labels'] for entry in batch]
    meta = [entry['meta'] for entry in batch]

    result_inp = {}
    for key in inputs[0]:
        result_inp[key] = [entry[key] for entry in inputs]
    result_label = {}
    for key in labels[0]:
        result_label[key] = [entry[key] for entry in labels]
    result_meta = {}
    for key in meta[0]:
        result_meta[key] = [entry[key] for entry in meta]
    return {'inputs': result_inp, 'labels': result_label, 'meta': result_meta}


def build_dataset(params, test=False):
    splits = params['data']['data_splits']
    datasets = {split:FGSceneDataset(split, params, test) for split in splits}
    return datasets