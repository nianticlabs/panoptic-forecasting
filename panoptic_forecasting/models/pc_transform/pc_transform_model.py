# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
from torch import nn
import torch.nn.functional as F
try:
    import torch_scatter
except:
    print("WARNING: cannot do point cloud transformation without torch_scatter library.")

from panoptic_forecasting.models.base_model import BaseModel


class PCTransformModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        self.ind = params['model'].get('only_this_ind')
        self.is_img = params['model'].get('is_img')
        self.debug = params['model'].get('debug')
        pass

    def predict(self, inputs, labels):
        K = inputs['intrinsics']
        extrinsics = inputs['extrinsics']
        depths = inputs['depth']
        depth_mask = inputs['depth_mask']
        target_T = inputs['target_T']
        segs = inputs['seg']
        if self.ind is not None:
            depths = depths[:, self.ind:self.ind+1]
            depth_mask = depth_mask[:, self.ind:self.ind+1]
            target_T = target_T[:, self.ind:self.ind+1]
            segs = segs[:, self.ind:self.ind+1]

        # Step 1: back project 2d points into 3D using formula:
        # pts3d = depths * K^-1 * pts2d
        b, inp_t, height, width = depths.shape
        vs, us = torch.meshgrid(
            torch.arange(height, dtype=torch.float, device=depths.device),
            torch.arange(width, dtype=torch.float, device=depths.device),
        )

        pts2d = torch.cat([
            us.reshape(-1, 1), vs.reshape(-1, 1),
            torch.ones(height*width, 1, dtype=torch.float, device=us.device)
        ], dim=-1).unsqueeze(0).expand(b, -1, -1)
        K_inv = torch.inverse(K).reshape(b, 1, 3, 3)

        # [b, 1, 3, 3] x [b, hw, 3, 1]. After squeeze result is [b, hw, 3]
        pts3d_c = (K_inv @ pts2d.unsqueeze(-1)).squeeze(-1)
        pts3d_c = pts3d_c.unsqueeze(1) * depths.reshape(b, inp_t, -1, 1)
        pts3d_c = torch.cat(
            [pts3d_c, torch.ones(b, inp_t, height*width, 1, device=K.device)],
            dim=-1
        )

        # Step 2: convert camera points (in RDF) to vehicle points (in FLU)
        # Here, pts3d is [b, inp_t, h*w, 4]
        pts3d_v = extrinsics.view(b, 1, 1, 4, 4) @ pts3d_c.unsqueeze(-1)

        # Step 3: transform points such that they lie in the final frame's
        # vehicle coordinate system
        # target_T shape: [b, inp_t, 4, 4]
        result_pts3d_v = target_T.unsqueeze(2) @ pts3d_v

        # Step 4: Project points to 2d (by first transforming to camera coordinates)
        result_pts3d_c = torch.inverse(extrinsics).reshape(b, 1, 1, 4, 4) @ result_pts3d_v
        result_pts3d_c = result_pts3d_c[:, :, :, :3] / result_pts3d_c[:, :, :, 3:4]
        result_depths = result_pts3d_c[:, :, :, 2]
        result2d = K.view(b, 1, 1, 3, 3) @ result_pts3d_c
        result2d = result2d[:, :, :, :2] / result2d[:, :, :, 2:3]
        #result2d = result2d.squeeze(-1).round().long()

        result2d = result2d.squeeze(-1)
        # Valid points have the following properties:
        # - They correspond to valid input depth values
        # - the depth values are > 0 (i.e. they lie in front of the camera)
        # - The u/v coordinates lie within the image
        inbounds_mask = (result2d[:, :, :, 0] >= 0) & \
                        (result2d[:, :, :, 0] < width) & \
                        (result2d[:, :, :, 1] >= 0) & \
                        (result2d[:, :, :, 1] < height)
        result_mask = depth_mask.view(b, inp_t, height*width)* \
                    (result_depths.squeeze(-1) > 0) & \
                      inbounds_mask

        # We need to translate our 2d predictions (which currently take the form
        # [batch, num_predicted_points, 2] and represent the u/v coordinates for each
        # point in the final camera frame) to the actual image, only keeping points
        # with valid depths and moreover keeping the closest valid point.

        # We do this using scatter (a good overview of how this works can be seen at
        # https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter)

        # First: find the points with the smallest depth at each result location
        result_mask = result_mask.reshape(b, inp_t*height*width)
        result_depths = result_depths.reshape(b, inp_t*height*width)

        # Make sure we never select an invalid point for a location when a
        # valid point exists
        result_depths[~result_mask] = result_depths.max() + 1
        result2d = result2d.reshape(b, inp_t*height*width, 2)
        result2d_0 = torch.stack([result2d[:, :, 0].floor().long(), result2d[:, :, 1].floor().long()], dim=-1)
        result2d_1 = torch.stack([result2d[:, :, 0].floor().long(), result2d[:, :, 1].ceil().long()], dim=-1)
        result2d_2 = torch.stack([result2d[:, :, 0].ceil().long(), result2d[:, :, 1].floor().long()], dim=-1)
        result2d_3 = torch.stack([result2d[:, :, 0].ceil().long(), result2d[:, :, 1].ceil().long()], dim=-1)

        result2d = torch.cat([result2d_0, result2d_1, result2d_2, result2d_3], dim=1)
        result2d[:, :, 0].clamp_(0, width-1)
        result2d[:, :, 1].clamp_(0, height-1)
        result_depths = result_depths.repeat(1, 4)

        scatter_inds = result2d[:, :, 1]*width + result2d[:, :, 0]
        _, argmin = torch_scatter.scatter_min(result_depths, scatter_inds, -1,
                                              dim_size=inp_t*height*width*4)
        tmp_mask = (argmin < inp_t*height*width*4)
        ind0 = tmp_mask.nonzero()[:, 0]
        ind1 = argmin[tmp_mask]
        tgt_ind1 = tmp_mask.nonzero()[:, 1]

        if self.is_img:
            final_seg = torch.zeros(b, height*width, 3, dtype=segs.dtype, device=K.device)
            segs = segs.reshape(b, inp_t*height*width, 3).repeat(1, 4, 1)
        else:
            final_seg = torch.zeros(b, height*width, dtype=segs.dtype, device=K.device)
            segs = segs.reshape(b, inp_t*height*width).repeat(1, 4)

        # The following ensures we don't copy a prediction from an "invalid" point
        segs[~result_mask.repeat(1,4)] = 0
        final_seg[ind0, tgt_ind1] = segs[ind0, ind1]

        final_depths = torch.zeros(b, height*width,
                                   dtype=result_depths.dtype,
                                   device=K.device).fill_(-1)
        final_depths[ind0, tgt_ind1] = result_depths[ind0, ind1]
        if self.is_img:
            final_seg = final_seg.view(b, height, width, 3)
        else:
            final_seg = final_seg.view(b, height, width)

        result_dict = {
            'seg': final_seg,
            'result2d': result2d[:, :inp_t*height*width].reshape(b, inp_t, height, width, 2),
            'depth': final_depths.view(b, height, width),
        }
        return result_dict
