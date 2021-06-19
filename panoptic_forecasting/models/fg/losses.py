# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
from torch import nn
import torch.nn.functional as F

from panoptic_forecasting.data import data_utils
from . import model_utils

class TrajectoryLoss():

    def __init__(self, 
                 loss_type,
                 use_depth_inp=False, 
                 use_bbox_ulbr=False,
                 supervise_last_inp=True,
                 only_loc_feats=False):
        super().__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'smoothl1':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError('loss_type not recognized:', loss_type)
        self.use_depth_inp = use_depth_inp
        self.use_bbox_ulbr = use_bbox_ulbr
        self.supervise_last_inp = supervise_last_inp
        self.only_loc_feats = only_loc_feats

    def __call__(self, inputs, labels, pred_dict):
        bbox_masks = inputs['bbox_masks']
        bbox_vel_masks = inputs['bbox_vel_masks']
        input_depths = inputs.get('depths')
        input_depth_masks = inputs.get('depth_masks')
        label_depths = labels.get('depths')
        label_depth_masks = labels.get('depth_masks')

        input_trajs = inputs['trajectories']
        n_input_trajs = inputs['normalized_trajectories']
        label_trajs = labels['trajectories']
        n_label_trajs = labels['normalized_trajectories']
        if isinstance(bbox_masks, list) or isinstance(bbox_masks, tuple):
            bbox_masks = torch.cat(bbox_masks)
            bbox_vel_masks = torch.cat(bbox_vel_masks)
            input_depths = torch.cat(input_depths)
            input_depth_masks = torch.cat(input_depth_masks)
            label_depths = torch.cat(label_depths)
            label_depth_masks = torch.cat(label_depth_masks)
            input_trajs = torch.cat(input_trajs)
            label_trajs = torch.cat(label_trajs)
            #n_input_trajs = torch.cat(n_input_trajs)
            #n_label_trajs = torch.cat(n_label_trajs)
        bbox_masks = bbox_masks.float()
        bbox_vel_masks = bbox_vel_masks.float()
        inp_t = input_trajs.size(1)
        out_t = label_trajs.size(1)
        traj_preds = pred_dict['normalized_trajectory']
        unnorm_traj_preds = pred_dict['unnormalized_trajectory']
        if 'normalized_depth' in pred_dict:
            traj_preds = torch.cat([traj_preds, pred_dict['normalized_depth']], dim=-1)
            unnorm_traj_preds = torch.cat([unnorm_traj_preds, pred_dict['unnormalized_depth']],dim=-1)


        traj_masks = model_utils.expand_traj_mask(bbox_masks, vel_mask=bbox_vel_masks)
        if self.supervise_last_inp:
            gt_trajs = torch.cat([input_trajs[:, -1].unsqueeze(1),
                                  label_trajs], dim=1)
            normalized_gt = torch.cat([n_input_trajs[:, -1].unsqueeze(1),
                                       n_label_trajs], dim=1)
            traj_masks = traj_masks[:, -out_t-1:]
        else:
            gt_trajs = label_trajs
            normalized_gt = n_label_trajs
            traj_masks = traj_masks[:, -out_t:]
        if self.only_loc_feats:
            gt_trajs = gt_trajs[:, :, :4]
            normalized_gt = normalized_gt[:, :, :4]
            traj_masks = traj_masks[:, :, :4]
        if self.use_depth_inp:
            if self.supervise_last_inp:
                gt_depths = torch.cat([input_depths[:, -1].unsqueeze(1),
                                       label_depths], dim=1)
            else:
                gt_depths = label_depths
            gt_depth_masks = torch.cat([input_depth_masks, label_depth_masks], dim=1).float().squeeze(-1)
            gt_depth_masks = model_utils.expand_traj_mask(gt_depth_masks,
                                                    result_size=1)
            if self.supervise_last_inp:
                gt_depth_masks = gt_depth_masks[:, -out_t-1:]
            else:
                gt_depth_masks = gt_depth_masks[:, -out_t:]
            if self.only_loc_feats:
                gt_depths = gt_depths[:, :, :1]
                gt_depth_masks = gt_depth_masks[:, :, :1]
            gt_trajs = torch.cat([gt_trajs, gt_depths], -1)
            traj_masks = torch.cat([traj_masks, gt_depth_masks], -1)

        loss_preds = unnorm_traj_preds
        loss_gt = gt_trajs

        
        traj_loss = self.loss_fn(loss_preds, loss_gt) * traj_masks
        traj_loss = traj_loss.view(traj_loss.size(0), -1).sum(-1) / (traj_masks.reshape(
            traj_masks.size(0), -1).sum(-1) + 1e-8)

        if self.use_depth_inp:
            if self.only_loc_feats:
                depth_preds = unnorm_traj_preds[:, :, 4:5]
            else:
                depth_preds = unnorm_traj_preds[:, :, 8:9]
        if self.use_bbox_ulbr:
            unnorm_traj_preds = data_utils.convert_bbox_ulbr_cwh(unnorm_traj_preds[:, :, :4])
            gt_trajs = data_utils.convert_bbox_ulbr_cwh(gt_trajs[:, :, :4])

        center_l2 = torch.norm(unnorm_traj_preds[:, :, :2] - gt_trajs[:, :, :2], dim=-1)
        if self.supervise_last_inp:
            bbox_masks = bbox_masks[:, -out_t - 1:]
        else:
            bbox_masks = bbox_masks[:, -out_t:]
        center_l2 = (center_l2 * bbox_masks).view(
            center_l2.size(0), -1).sum(-1) / (bbox_masks.view(bbox_masks.size(0), -1).sum(-1) + 1e-8)

        fde = torch.norm(unnorm_traj_preds[:, -1, :2] - gt_trajs[:, -1, :2], dim=-1)
        fde = (fde * bbox_masks[:, -1]).view(
            fde.size(0), -1).sum(-1)

        size_l1 = F.l1_loss(unnorm_traj_preds[:, :, 2:4], gt_trajs[:, :, 2:4], reduction='none')
        size_l1 = (size_l1 * bbox_masks.unsqueeze(-1)).view(
            size_l1.size(0), -1).sum(-1) / (bbox_masks.sum(-1) + 1e-8)

        final_result = {
            'traj_2d_loss': traj_loss,
            'center_pixel_l2': center_l2,
            'center_pixel_fde': fde,
            'size_pixel_l1': size_l1,
        }
        if self.use_depth_inp:
            depth_l2 = torch.norm(depth_preds - gt_depths[:, :, 0:1], dim=-1)
            tmp_depth_masks = gt_depth_masks[:, :, 0]
            div = tmp_depth_masks.sum(-1)
            div[div == 0] = 1
            depth_l2 = (depth_l2 * tmp_depth_masks).sum(-1) / div
            final_result['depth_l2'] = depth_l2
        return traj_loss, final_result


class DefaultMaskLoss():
    def __init__(self, mask_distill_coef=1.0,
                 supervise_last_inp=True):
        self.mask_distill_coef = mask_distill_coef
        self.supervise_last_inp = supervise_last_inp

    def __call__(self, inputs, labels, pred_dict):
        feat_masks = inputs['feat_masks']
        inp_feats = inputs['feats']
        label_feats = labels['feats']
        if isinstance(feat_masks, list) or isinstance(feat_masks, tuple):
            feat_masks = torch.cat(feat_masks)
            inp_feats = torch.cat(inp_feats)
            label_feats = torch.cat(label_feats)
        inp_t = inp_feats.size(1)
        out_t = label_feats.size(1)
        if self.supervise_last_inp:
            feat_masks = feat_masks[:, -out_t - 1:]
        else:
            feat_masks = feat_masks[:, -out_t:]
        mask_feat_preds = pred_dict['mask_feats']
        if self.supervise_last_inp:
            target_feats = torch.cat([
                inp_feats[:, -1].unsqueeze(1),
                label_feats,
            ], dim=1)
        else:
            target_feats = label_feats
        distill_loss = F.mse_loss(mask_feat_preds, target_feats, reduction='none')
        b, t, c, h, w = distill_loss.shape
        
        distill_loss = distill_loss.view(b, t, -1).sum(-1) * feat_masks
        distill_loss = distill_loss.sum(-1) / ((feat_masks.sum(-1) * c * h * w)+1e-8)

        loss_dict = {
            'mask_distill_loss': distill_loss,
        }

        return distill_loss, loss_dict
