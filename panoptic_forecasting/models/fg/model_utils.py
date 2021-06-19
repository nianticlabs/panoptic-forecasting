# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn.functional as F


def expand_traj_mask(mask, vel_mask=None, first_mask=None, result_size=4, no_vel=False):
    if first_mask is None:
        loc_mask = mask
    else:
        loc_mask = mask * (1 - first_mask.squeeze(-1))
    if no_vel:
        return loc_mask.unsqueeze(-1).expand(-1, -1, result_size)
    if vel_mask is None:
        vel_mask = torch.cat([
            torch.zeros(mask.size(0), 1, device=mask.device),
            mask[:, 1:] * mask[:, :-1]
        ], dim=1)
    result = torch.cat([
        loc_mask.unsqueeze(-1).expand(-1, -1, result_size),
        vel_mask.unsqueeze(-1).expand(-1, -1, result_size),
    ], dim=-1)
    return result

# based on https://github.com/facebookresearch/detectron2/blob/3a1d2fde77c7697a64377a8a15269687228cea42/detectron2/layers/mask_ops.py#L16
def paste_mask(masks, bbox, img_h, img_w, use_bbox_ulbr,
               fill_value=None):
    N = masks.shape[0]
    if use_bbox_ulbr:
        x0, y0, x1, y1 = torch.split(bbox, 1, dim=1)
    else:
        cx, cy, w, h = torch.split(bbox, 1, dim=1)
        x0 = (cx - w / 2)
        x1 = (cx + w / 2)
        y0 = (cy - h / 2)
        y1 = (cy + h / 2)
    img_y = torch.arange(0, img_h, device=masks.device, dtype=torch.float32) + 0.5
    img_x = torch.arange(0, img_w, device=masks.device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)
    img_masks = F.grid_sample(masks.to(dtype=torch.float32, device=masks.device), grid,
                              align_corners=False)
    if fill_value is not None:
        img_masks = img_masks.clone()
        img_masks[img_masks == 0] = fill_value
    if img_masks.size(1) == 0:
        return img_masks[:, 0]
    else:
        return img_masks
