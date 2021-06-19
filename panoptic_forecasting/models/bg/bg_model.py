# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import hardnet
from panoptic_forecasting.models.base_model import BaseModel


class BGModel(BaseModel):

    def __init__(self, params):
        super().__init__()
        self.num_classes = num_classes = params['data']['num_classes']
        self.use_depth_inps = params['model'].get('use_depth_inps')
        self.num_inputs = params['model'].get('num_inputs', 1)
        self.min_depth = params['data'].get('min_depth')
        self.max_depth = params['data'].get('max_depth')
        self.convert2onehot = params['model'].get('convert2onehot')
        
        final_w = params['model'].get('final_w')
        final_h = params['model'].get('final_h')
        if final_w is not None and final_h is not None:
            self.final_size = (final_h, final_w)
        else:
            self.final_size = None
        if self.use_depth_inps:
            depth_size = 1
            depth_norm_params = params['data'].get('depth_norm_params')
            if depth_norm_params is None:
                mean = torch.zeros(1)
                std = torch.zeros(1)
            else:
                mean, std = depth_norm_params
            self.depth_mean = nn.Parameter(mean, requires_grad=False)
            self.depth_std = nn.Parameter(std, requires_grad=False)
            num_classes += depth_size
        num_classes *= self.num_inputs
        self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        self.model = hardnet.build_hardnet(params)

        # Need to modify so it accepts 19-channel input
        self.model.expand_first_layer(num_classes)

    def _normalize_depths(self, depths):
        return (depths - self.depth_mean)/self.depth_std

    def _inp2onehot(self, inp):
        mask = inp < self.num_classes
        inp[~mask] = 0
        result = F.one_hot(inp, num_classes=self.num_classes)
        result *= mask.unsqueeze(-1)
        result = result.permute(0, 1,4, 2, 3).float()
        return result

    def forward(self, inps, depths, depth_masks, return_orig_size=False):
        if self.convert2onehot:
            inps = self._inp2onehot(inps)
        b, t, c, h, w = inps.shape
        inps = inps.reshape(b, t*c, h, w)
        if self.use_depth_inps:
            depths = self._normalize_depths(depths)
            depths = depths * depth_masks
            inps = torch.cat([inps, depths], dim=1)
        preds = self.model(inps, final_size=self.final_size, return_orig_size=return_orig_size)
        return preds

    def loss(self, inputs, labels):
        inps = inputs['seg']
        seg_labels = labels['seg']
        depths = inputs.get('depth')
        depth_masks = inputs.get('depth_mask')

        seg_preds = self(inps, depths, depth_masks)

        seg_loss = self.seg_loss_fn(seg_preds, seg_labels)
        max_preds = seg_preds.argmax(1).long()
        correct = (max_preds == seg_labels).sum()
        total = (seg_labels != 255).sum()
        final_result = {
            'loss': seg_loss,
            'accuracy': correct.float() / total.float(),
        }
        return final_result

    def predict(self, inputs, labels):
        inps = inputs['seg']
        depths = inputs.get('depth')
        depth_masks = inputs.get('depth_mask')
        logits = self(inps, depths, depth_masks, return_orig_size=True)
        logits, orig_size_logits = logits
        final_result = {}
        preds = logits.argmax(1)
        final_result['seg'] = preds
        final_result['logits'] = logits
        final_result['orig_size_logits'] = orig_size_logits
        return final_result