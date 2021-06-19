# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from panoptic_forecasting.models.base_model import BaseModel
from panoptic_forecasting.data import data_utils
from .mask_rcnn_conv_upsample_head import MaskRCNNConvUpsampleHead
from . import losses
from . import model_utils
from .convlstm import ConvLSTM


class FGModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        self.supervise_last_inp = True
        self.mask_distill_coef = params['model'].get('mask_distill_coef', 1.0)
        self.traj_coef = params['model'].get('traj_coef', 1)
        rnn_hidden = params['model']['rnn_hidden']
        loss_type = params['model']['loss_type']
        instance_feat_channels = params['model'].get('instance_feat_channels')
        traj_feat_channels = params['model'].get('traj_feat_channels')
        instance_feat_hidden = params['model'].get('instance_feat_hidden')
        self.use_odometry = params['model'].get('use_odometry')
        self.only_input_odometry = params['model'].get('only_input_odometry')
        self.use_bbox_ulbr = params.get('use_bbox_ulbr')
        self.rnn_type = params['model'].get('rnn_type')
        self.use_depth_inp = params['model'].get('use_depth_inp')
        self.use_depth_sorting = params['model'].get('use_depth_sorting')
        self.mask_loss_type = params['model'].get('mask_loss_type', 'default')
        #TODO: I removed use_background_in_instances. double check that this was the right move
        self.only_loc_feats = params['model'].get('only_loc_feats')

        self.no_traj_inst_feats = params['model'].get('no_traj_inst_feats')
        self.no_mask_traj_feats = params['model'].get('no_mask_traj_feats')
        num_traj_out_layers = params['model'].get('num_traj_out_layers', 1)
        num_convlstm_layers = params['model'].get('num_convlstm_layers', 1)

        if self.rnn_type == 'lstm':
            rnn_mod = nn.LSTM
        elif self.rnn_type == 'gru':
            rnn_mod = nn.GRU
        else:
            raise ValueError('rnn_type not recognized: ',self.rnn_type)

        if self.only_loc_feats:
            encoder_inp = 5
            out_size = 4
        else:
            encoder_inp = 9
            out_size = 8


        if self.use_odometry:
            odom_size = params['data']['odom_size']
            encoder_inp += odom_size
            odom_norm_params = params['data'].get('odom_norm_params')
            if odom_norm_params is None:
                mean = torch.zeros(odom_size)
                std = torch.zeros(odom_size)
            else:
                mean, std = odom_norm_params
            print("ODOM NORM PARAMS: ",mean, std)
            self.odom_mean = nn.Parameter(mean.unsqueeze(0), requires_grad=False)
            self.odom_std = nn.Parameter(std.unsqueeze(0), requires_grad=False)
        if self.use_depth_inp:
            if self.only_loc_feats:
                encoder_inp += 1
                out_size += 1
            else:
                encoder_inp += 2
                out_size += 2
            depth_norm_params = params['data'].get('depth_norm_params')
            if depth_norm_params is None:
                mean = torch.zeros(2)
                std = torch.zeros(2)
            else:
                mean, std = depth_norm_params
            if self.only_loc_feats:
                mean = mean[:1]
                std = std[:1]
            self.depth_mean = nn.Parameter(mean.unsqueeze(0), requires_grad=False)
            self.depth_std = nn.Parameter(std.unsqueeze(0), requires_grad=False)
        decoder_inp = encoder_inp - 1
        if self.use_odometry and self.only_input_odometry:
            decoder_inp -= odom_size
        if self.no_traj_inst_feats:
            traj_encoder_inp = encoder_inp
            traj_decoder_inp = decoder_inp
        else:
            traj_encoder_inp = encoder_inp + instance_feat_hidden
            traj_decoder_inp = decoder_inp + instance_feat_hidden
        self.traj_encoder = rnn_mod(traj_encoder_inp,
                                    rnn_hidden, batch_first=True)
        self.traj_decoder = rnn_mod(traj_decoder_inp,
                                    rnn_hidden, batch_first=True)

        norm_params = params['data'].get('norm_params')
        if norm_params is None:
            mean = torch.zeros(8)
            std = torch.zeros(8)
        else:
            mean, std = params['data']['norm_params']
        if self.only_loc_feats:
            mean = mean[:4]
            std = std[:4]
        self.traj_mean = nn.Parameter(mean.unsqueeze(0), requires_grad=False)
        self.traj_std = nn.Parameter(std.unsqueeze(0), requires_grad=False)

        if num_traj_out_layers == 1:
            self.traj_encoder_out = nn.Linear(rnn_hidden, out_size)
            self.traj_decoder_out = nn.Linear(rnn_hidden, out_size)
        else:
            decoder_out_layers = []
            encoder_out_layers = []
            for _ in range(num_traj_out_layers-1):
                encoder_out_layers.append(nn.Linear(rnn_hidden, rnn_hidden))
                encoder_out_layers.append(nn.ReLU(inplace=True))
                decoder_out_layers.append(nn.Linear(rnn_hidden, rnn_hidden))
                decoder_out_layers.append(nn.ReLU(inplace=True))
            encoder_out_layers.append(nn.Linear(rnn_hidden, out_size))
            decoder_out_layers.append(nn.Linear(rnn_hidden, out_size))
            self.traj_encoder_out = nn.Sequential(*encoder_out_layers)
            self.traj_decoder_out = nn.Sequential(*decoder_out_layers)
        traj_feat_out_size = traj_feat_channels
        self.traj_feat_out = nn.Linear(rnn_hidden, traj_feat_out_size)

        self.instance_compressor = nn.Conv2d(256, instance_feat_channels, (1,1))
        self.instance_feat_model = nn.Linear(instance_feat_channels*14*14, instance_feat_hidden)
        if self.no_mask_traj_feats:
            feat_inp_size = 256
        else:
            feat_inp_size = 256 + traj_feat_channels
        self.mask_encoder = ConvLSTM(
            feat_inp_size, 256, (3,3), num_convlstm_layers, batch_first=True,
            return_all_layers=True,
        )
        self.mask_decoder = ConvLSTM(
            feat_inp_size, 256, (3,3), num_convlstm_layers, batch_first=True,
            return_all_layers=True,
        )
        self.mask_encoder_out = nn.Conv2d(256, 256, (1,1), 1)
        self.mask_decoder_out = nn.Conv2d(256, 256, (1,1), 1)
        self.mask_head = MaskRCNNConvUpsampleHead(params)

        self.traj_loss = losses.TrajectoryLoss(
            loss_type,
            use_depth_inp = self.use_depth_inp,
            use_bbox_ulbr=self.use_bbox_ulbr,
            supervise_last_inp=self.supervise_last_inp,
            only_loc_feats=self.only_loc_feats,
        )
        self.mask_loss = losses.DefaultMaskLoss(
            mask_distill_coef=self.mask_distill_coef,
            supervise_last_inp=self.supervise_last_inp,
        )

        #TODO: YOU ARE HERE
    def _normalize(self, inp, mean, std):
        old_shape = inp.shape
        inp = inp.reshape(-1, old_shape[-1])
        normalized_inp = (inp - mean) / std
        return normalized_inp.reshape(old_shape)

    def _unnormalize(self, inp, mean, std):
        old_shape = inp.shape
        inp = inp.reshape(-1, old_shape[-1])
        result = inp * std + mean
        return result.reshape(old_shape)

    def _normalize_traj(self, trajs, depths):
        traj_mean = self.traj_mean
        traj_std = self.traj_std
        if self.use_depth_inp:
            trajs = torch.cat([trajs, depths], dim=-1)
            mean = torch.cat([traj_mean, self.depth_mean.expand(len(traj_mean), -1)], -1)
            std = torch.cat([traj_std, self.depth_std.expand(len(traj_mean), -1)], -1)
            return self._normalize(trajs, mean, std)
        else:
            return self._normalize(trajs, traj_mean, traj_std)

    def _unnormalize_traj(self, trajs):
        traj_mean = self.traj_mean
        traj_std = self.traj_std
        if self.use_depth_inp:
            mean = torch.cat([traj_mean, self.depth_mean.expand(len(traj_mean), -1)], -1)
            std = torch.cat([traj_std, self.depth_std.expand(len(traj_mean), -1)], -1)
            return self._unnormalize(trajs, mean, std)
        else:
            return self._unnormalize(trajs, traj_mean, traj_std)

    def _normalize_odom(self, odom):
        return self._normalize(odom, self.odom_mean, self.odom_std)

    def _unnormalize_odom(self, odom):
        return self._unnormalize(odom, self.odom_mean, self.odom_std)

    def _compute_traj_inst_feats(self, inst_feats, mask):
        b,t = inst_feats.shape[:2]
        inst_feats = inst_feats.reshape(-1, 256, 14, 14)
        x = self.instance_compressor(inst_feats)
        x = F.relu(x.view(x.size(0), -1))
        x = self.instance_feat_model(x)
        x = x.view(b, t, -1)
        x = x * mask
        return x

    def forward(self, input_trajs, traj_mask, traj_vel_mask, instance_feats, output_inds,
                odom, input_depths, input_depth_masks, classes,
                num_output_steps):
        ##################################
        # First part: get the input set up
        ##################################
        b = input_trajs.size(0)
        traj_mask = traj_mask.float()
        traj_vel_mask = traj_vel_mask.float()
        if self.only_loc_feats:
            input_trajs = input_trajs[:, :, :4]
        if self.only_loc_feats and input_depths is not None:
            input_depths = input_depths[:, :, :1]
        normalized_inps = self._normalize_traj(input_trajs, input_depths)
        if self.use_depth_inp:
            input_trajs = torch.cat([input_trajs, input_depths], -1)
        expanded_traj_mask = model_utils.expand_traj_mask(traj_mask, vel_mask=traj_vel_mask,
                                                          no_vel=self.only_loc_feats)
        if self.use_depth_inp:
            input_depth_masks = input_depth_masks.float().squeeze(-1)
            expanded_depth_mask = model_utils.expand_traj_mask(input_depth_masks,
                                                          result_size=1, no_vel=self.only_loc_feats)
            expanded_traj_mask = torch.cat([expanded_traj_mask,
                                            expanded_depth_mask], -1)
        normalized_inps *= expanded_traj_mask
        if self.use_odometry:
            odom = self._normalize_odom(odom)

        traj_mask = traj_mask.unsqueeze(-1)
        traj_inp_mask = traj_mask
        ones_mask = torch.ones(input_trajs.size(0), 1, 1,
                                device=input_trajs.device)

        #######################################
        # Second part: run the encoder
        #######################################
        if not self.no_traj_inst_feats:
            encoder_traj_inst_feats = self._compute_traj_inst_feats(instance_feats,
                                                                    traj_mask)
        encoder_inp = [normalized_inps]
        if not self.no_traj_inst_feats:
            encoder_inp.append(encoder_traj_inst_feats)
        encoder_inp.append(traj_inp_mask)
        inp_t = input_trajs.size(1)
        if self.use_odometry:
            encoder_inp.append(odom[:, :inp_t])
        encoder_inp = torch.cat(encoder_inp, -1)
        traj_encoder_result, traj_encoder_state = self.traj_encoder(encoder_inp)
        if not self.no_mask_traj_feats:
            mask_encoder_feats = self.traj_feat_out(traj_encoder_result).unsqueeze(-1).unsqueeze(
                -1).expand(-1, -1, -1, 14, 14)

            mask_inp = torch.cat([mask_encoder_feats, instance_feats], dim=2)
        else:
            mask_inp = instance_feats
        mask_encoder_result, mask_encoder_state = self.mask_encoder(mask_inp)

        ################################################
        # Third part: predict bbox features and instance features
        #             at most recent input frame
        #################################################
        # Note: this step is done because we do not always have
        # input at the most recent frame (due to, e.g., occlusions)
        current_traj = self.traj_encoder_out(traj_encoder_result[:, -1].unsqueeze(1))
        current_mask_feats = self.mask_encoder_out(
            mask_encoder_result[-1][:, -1]
        ).unsqueeze(1)
        current_inp_traj = current_traj

        ################################################
        # Fourth part: run the decoder
        ################################################
        traj_preds = [current_traj]
        mask_feat_preds = [current_mask_feats]
        traj_decoder_state = traj_encoder_state
        mask_decoder_state = mask_encoder_state
        out_odom = odom[:, inp_t:]
        out_t = num_output_steps
        for t in range(out_t):
            # predictions for trajectories
            if not self.no_traj_inst_feats:
                tmp_inps = current_mask_feats
                traj_inst_feats = self._compute_traj_inst_feats(tmp_inps, ones_mask)
            current_inp = [current_inp_traj]
            if not self.no_traj_inst_feats:
                current_inp.append(traj_inst_feats)
            if self.use_odometry and not self.only_input_odometry:
                current_inp.append(out_odom[:, t:t+1])
            current_inp = torch.cat(current_inp, -1)
            traj_decoder_result, traj_decoder_state = self.traj_decoder(current_inp,
                                                           traj_decoder_state)

            traj_decoder_out = self.traj_decoder_out(traj_decoder_result)
            current_traj =  current_traj + traj_decoder_out
            traj_preds.append(current_traj)
            current_inp_traj = current_traj

            # predictions for mask
            if not self.no_mask_traj_feats:
                mask_decoder_out = self.traj_feat_out(traj_decoder_result).unsqueeze(-1).unsqueeze(
                    -1).expand(-1, -1, -1, 14, 14)
                mask_inp = torch.cat([mask_decoder_out, current_mask_feats], dim=2)
            else:
                mask_inp = current_mask_feats
            mask_decoder_result, mask_decoder_state = self.mask_decoder(mask_inp,
                                                                        mask_decoder_state)
            current_mask_feats = self.mask_decoder_out(mask_decoder_result[-1].squeeze(1))
            current_mask_feats.unsqueeze_(1)
            mask_feat_preds.append(current_mask_feats)

        traj_preds = torch.cat(traj_preds, dim=1)

        mask_feat_preds = torch.cat(mask_feat_preds, dim=1)
        output_feats = mask_feat_preds[:, -num_output_steps:][range(b), output_inds]
        mask_preds = self.mask_head(output_feats)
        mask_preds = mask_preds[range(b), classes]
        unnorm_traj_preds = self._unnormalize_traj(traj_preds)
        return {
            'normalized_trajectory': traj_preds,
            'unnormalized_trajectory': unnorm_traj_preds,
            'mask_feats': mask_feat_preds,
            'output_feats': output_feats,
            'masks': mask_preds,
        }

    def loss(self, inputs, labels):
        input_trajs = inputs['trajectories']
        label_trajs = labels['trajectories']
        input_depths = inputs.get('depths')
        label_depths = labels.get('depths')
        input_depth_masks = inputs.get('depth_masks')
        bbox_masks = inputs['bbox_masks'].float()
        bbox_vel_masks = inputs['bbox_vel_masks'].float()
        odom = inputs.get('odometry')
        input_feats = inputs['feats']
        output_inds = labels['output_inds']
        classes = inputs['classes']

        inp_t = input_trajs.size(1)
        out_t = label_trajs.size(1)
        pred_inp_trajs = input_trajs
        pred_dict = self(
            pred_inp_trajs, bbox_masks[:, :inp_t], bbox_vel_masks[:, :inp_t], input_feats,
            output_inds, odom, input_depths, input_depth_masks,
            classes, out_t,
        )
        if self.only_loc_feats:
            input_trajs = input_trajs[:, :, :4]
            label_trajs = label_trajs[:, :, :4]
            if input_depths is not None:
                input_depths = input_depths[:, :, :1]
                label_depths = label_depths[:, :, :1]

        inputs['normalized_trajectories'] = self._normalize_traj(input_trajs,
                                                                 input_depths)
        labels['normalized_trajectories'] = self._normalize_traj(label_trajs,
                                                                 label_depths)
        
        traj_loss, traj_loss_dict = self.traj_loss(inputs, labels, pred_dict)
        loss = self.traj_coef*traj_loss
        final_result = traj_loss_dict


        distill_loss, mask_loss_dict = self.mask_loss(
            inputs, labels, pred_dict,
        )
        loss = loss + self.mask_distill_coef * distill_loss
        final_result.update(mask_loss_dict)


        final_result['loss'] = loss
        return final_result

    def predict_semantics(self, inputs, labels):
        input_trajs = inputs['trajectories']
        label_trajs = labels['trajectories']
        input_depths = inputs.get('depths')
        input_depth_masks = inputs.get('depth_masks')
        if input_depths is not None:
            input_depths = torch.cat(input_depths)
            input_depth_masks = torch.cat(input_depth_masks)

        bbox_masks = inputs['bbox_masks']
        bbox_vel_masks = inputs['bbox_vel_masks']
        odom = inputs.get('odometry')
        input_feats = inputs['feats']
        output_inds = labels['output_inds']
        orig_classes = inputs['classes']
        b = len(input_feats)
        num_instances = torch.LongTensor([len(inst) for inst in input_feats])
        input_feats = torch.cat(input_feats)
        input_trajs = torch.cat(input_trajs)
        label_trajs = torch.cat(label_trajs)
        classes = torch.cat(orig_classes)
        output_inds = torch.cat(output_inds)
        bbox_masks = torch.cat(bbox_masks).float()
        bbox_vel_masks = torch.cat(bbox_vel_masks).float()
        if odom is not None:
            odom = torch.cat(odom)
        if 'background' in inputs:
            final_result = torch.stack(inputs['background'])
        else:
            final_result = torch.ones(b, 1024, 2048, device=input_feats.device)*255
        background_depths = inputs.get('background_depth')
        background_depth_masks = inputs.get('background_depth_mask')
        if background_depths is not None:
            background_depths = torch.stack(background_depths)
        if background_depth_masks is not None:
            background_depth_masks = torch.stack(background_depth_masks)

        out_t = label_trajs.size(1)

        pred_inp_trajs = input_trajs
        pred_dict = self(
            pred_inp_trajs, bbox_masks[:, :input_trajs.size(1)],
            bbox_vel_masks[:, :input_trajs.size(1)], input_feats,
            output_inds, odom, input_depths, input_depth_masks,
            classes, out_t,
        )
        traj_preds = pred_dict['unnormalized_trajectory']
        mask_preds = pred_dict['masks']
        mask_preds = torch.sigmoid(mask_preds)
        mask_preds = mask_preds.split(list(num_instances))
        traj_preds = traj_preds[:, -out_t:]

        full_traj_preds = traj_preds[:, :, :4].split(list(num_instances))

        pred_bboxes = traj_preds[range(len(traj_preds)), output_inds, :4]
        pred_bboxes = pred_bboxes.split(list(num_instances))
        if self.use_depth_inp:
            if self.only_loc_feats:
                all_pred_depths = traj_preds[range(len(traj_preds)), :, 4]
                pred_depths = traj_preds[range(len(traj_preds)), output_inds, 4]
            else:
                all_pred_depths = traj_preds[range(len(traj_preds)), :, 8]
                pred_depths = traj_preds[range(len(traj_preds)), output_inds, 8]
            pred_depths = pred_depths.split(list(num_instances))
            all_pred_depths = all_pred_depths.split(list(num_instances))
        for b_ind, (b_mask_preds, bboxes, orig_class) in enumerate(
            zip(mask_preds, pred_bboxes, orig_classes)
        ):
            if self.use_depth_sorting:
                seq_depths = pred_depths[b_ind]
                inst_depths, inst_order = seq_depths.sort(descending=True)
                if background_depths is not None:
                    current_depths = background_depths[b_ind]
                    if background_depth_masks is not None:
                        current_depths[~background_depth_masks[b_ind]] = 1000000000
            else:
                inst_order = range(len(bboxes))
            for inst_ind in inst_order:
                mask_pred, bbox = b_mask_preds[inst_ind:inst_ind+1], bboxes[inst_ind:inst_ind+1]
                pasted_mask = model_utils.paste_mask(mask_pred.unsqueeze(1), bbox, 1024, 2048,
                                                      self.use_bbox_ulbr)
                pasted_mask = (pasted_mask >= 0.5).long()*(orig_class[inst_ind]+11)
                pasted_mask = pasted_mask.squeeze(0)
                if self.use_depth_sorting and background_depths is not None:
                    inst_depth = seq_depths[inst_ind]
                    depth_mask = (inst_depth < current_depths)*(pasted_mask > 0)
                    final_result[b_ind] = (depth_mask)*pasted_mask + \
                                          (~(depth_mask))*final_result[b_ind]
                    current_depths[depth_mask] = inst_depth
                else:
                    final_result[b_ind] = (pasted_mask > 0)*pasted_mask + \
                                          (~(pasted_mask > 0))*final_result[b_ind]
        result_dict = {
            'seg': final_result,
            #'bbox': pred_bboxes,
            'bbox': full_traj_preds,
            'depths': all_pred_depths,
        }
        return result_dict

    def predict_panoptic(self, inputs, labels):
        input_trajs = inputs['trajectories']
        label_trajs = labels['trajectories']
        input_depths = inputs.get('depths')
        input_depth_masks = inputs.get('depth_masks')
        if input_depths is not None:
            input_depths = torch.cat(input_depths)
            input_depth_masks = torch.cat(input_depth_masks)

        bbox_masks = inputs['bbox_masks']
        bbox_vel_masks = inputs['bbox_vel_masks']
        odom = inputs.get('odometry')
        input_feats = inputs['feats']
        output_inds = labels['output_inds']
        orig_classes = inputs['classes']
        #gt_masks = labels['train_masks']
        b = len(input_feats)
        num_instances = torch.LongTensor([len(inst) for inst in input_feats])
        input_feats = torch.cat(input_feats)
        input_trajs = torch.cat(input_trajs)
        label_trajs = torch.cat(label_trajs)
        classes = torch.cat(orig_classes)
        output_inds = torch.cat(output_inds)
        bbox_masks = torch.cat(bbox_masks).float()
        bbox_vel_masks = torch.cat(bbox_vel_masks).float()
        if odom is not None:
            odom = torch.cat(odom)
        if 'background' in inputs:
            final_result = torch.stack(inputs['background'])
            final_result[final_result >= 11] = 255
        else:
            #final_result = torch.zeros(b, 1024, 2048, device=input_feats.device)
            final_result = torch.ones(b, 1024, 2048, device=input_feats.device)*255
        background_depths = inputs.get('background_depth')
        background_depth_masks = inputs.get('background_depth_mask')
        if background_depths is not None:
            background_depths = torch.stack(background_depths)
        if background_depth_masks is not None:
            background_depth_masks = torch.stack(background_depth_masks)

        out_t = label_trajs.size(1)

        pred_inp_trajs = input_trajs
        pred_dict = self(
            pred_inp_trajs, bbox_masks[:, :input_trajs.size(1)],
            bbox_vel_masks[:, :input_trajs.size(1)], input_feats,
            output_inds, odom, input_depths, input_depth_masks,
            classes, out_t, 
        )
        traj_preds = pred_dict['unnormalized_trajectory']
        mask_preds = pred_dict['masks']
        mask_preds = torch.sigmoid(mask_preds)
        mask_preds = mask_preds.split(list(num_instances))
        traj_preds = traj_preds[:, -out_t:]

        full_traj_preds = traj_preds[:, :, :4].split(list(num_instances))

        pred_bboxes = traj_preds[range(len(traj_preds)), output_inds, :4]
        pred_bboxes = pred_bboxes.split(list(num_instances))
        if self.use_depth_inp:
            if self.only_loc_feats:
                all_pred_depths = traj_preds[range(len(traj_preds)), :, 4]
                pred_depths = traj_preds[range(len(traj_preds)), output_inds, 4]
            else:
                all_pred_depths = traj_preds[range(len(traj_preds)), :, 8]
                pred_depths = traj_preds[range(len(traj_preds)), output_inds, 8]
            pred_depths = pred_depths.split(list(num_instances))
            all_pred_depths = all_pred_depths.split(list(num_instances))
        for b_ind, (b_mask_preds, bboxes, orig_class) in enumerate(
            zip(mask_preds, pred_bboxes, orig_classes)
        ):
            if self.use_depth_sorting:
                seq_depths = pred_depths[b_ind]
                inst_depths, inst_order = seq_depths.sort(descending=True)
                if background_depths is not None:
                    current_depths = background_depths[b_ind:b_ind+1]
                    if background_depth_masks is not None:
                        current_depths[~background_depth_masks[b_ind]] = 1000000000
            else:
                inst_order = range(len(bboxes))
            cl_ids = defaultdict(int)
            for inst_ind in inst_order:
                mask_pred, bbox = b_mask_preds[inst_ind:inst_ind+1], bboxes[inst_ind:inst_ind+1]
                pasted_mask = model_utils.paste_mask(mask_pred.unsqueeze(1), bbox, 1024, 2048,
                                                      self.use_bbox_ulbr)
                tmp_cl = orig_class[inst_ind].item()
                inst_id = cl_ids[tmp_cl]
                cl_ids[tmp_cl] += 1
                seg_val = (orig_class[inst_ind]+11)*1000 + inst_id
                pasted_mask = (pasted_mask >= 0.5).long()*seg_val
                pasted_mask = pasted_mask.squeeze(0)
                if self.use_depth_sorting and background_depths is not None:
                    inst_depth = seq_depths[inst_ind]
                    depth_mask = (inst_depth < current_depths)*(pasted_mask > 0)
                    final_result[b_ind] = (depth_mask)*pasted_mask + \
                                          (~(depth_mask))*final_result[b_ind]
                    current_depths[depth_mask] = inst_depth
                else:
                    final_result[b_ind] = (pasted_mask > 0)*pasted_mask + \
                                          (~(pasted_mask > 0))*final_result[b_ind]
        result_dict = {
            'seg': final_result,
            #'bbox': pred_bboxes,
            'bbox': full_traj_preds,
            'depths': all_pred_depths,
        }
        return result_dict

    def predict_instances(self, inputs, labels):
        input_trajs = inputs['trajectories']
        label_trajs = labels['trajectories']
        input_depths = inputs.get('depths')
        input_depth_masks = inputs.get('depth_masks')
        if input_depths is not None:
            input_depths = torch.cat(input_depths)
            input_depth_masks = torch.cat(input_depth_masks)

        bbox_masks = inputs['bbox_masks']
        bbox_vel_masks = inputs['bbox_vel_masks']
        odom = inputs.get('odometry')
        input_feats = inputs['feats']
        output_inds = labels['output_inds']
        orig_classes = inputs['classes']
        #gt_masks = labels['train_masks']
        inst_scores = inputs.get('inst_scores')
        b = len(input_feats)
        num_instances = torch.LongTensor([len(inst) for inst in input_feats])
        input_feats = torch.cat(input_feats)
        input_trajs = torch.cat(input_trajs)
        label_trajs = torch.cat(label_trajs)
        classes = torch.cat(orig_classes)
        output_inds = torch.cat(output_inds)
        bbox_masks = torch.cat(bbox_masks).float()
        bbox_vel_masks = torch.cat(bbox_vel_masks).float()
        if odom is not None:
            odom = torch.cat(odom)
        if 'background' in inputs:
            final_result = torch.stack(inputs['background'])
        else:
            final_result = torch.zeros(b, 1024, 2048, device=input_feats.device)

        background_depths = inputs.get('background_depth')
        background_depth_masks = inputs.get('background_depth_masks')
        if background_depths is not None:
            background_depths = torch.stack(background_depths)
        if background_depth_masks is not None:
            background_depth_masks = torch.stack(background_depth_masks)

        out_t = label_trajs.size(1)
        pred_inp_trajs = input_trajs
        pred_dict = self(
            pred_inp_trajs, bbox_masks[:, :input_trajs.size(1)],
            bbox_vel_masks[:, :input_trajs.size(1)], input_feats,
            output_inds, odom, input_depths, input_depth_masks,
            classes, out_t,
        )

        starting_seg = torch.zeros(b, 1024, 2048, device=input_feats.device)
        traj_preds = pred_dict['unnormalized_trajectory']
        mask_preds = pred_dict['masks']
        feat_preds = pred_dict['output_feats']
        mask_preds = torch.sigmoid(mask_preds)
        mask_preds = mask_preds.split(list(num_instances))
        traj_preds = traj_preds[:, -out_t:]


        full_traj_preds = traj_preds[:, :, :4].split(list(num_instances))

        pred_bboxes = traj_preds[range(len(traj_preds)), output_inds, :4]
        pred_bboxes = pred_bboxes.split(list(num_instances))
        feat_preds = feat_preds.split(list(num_instances))
        final_result = []
        final_classes = []
        final_logits = []
        final_logit_classes = []
        final_logit_bboxes = []
        final_feats = []
        final_unscaled_masks = []
        final_depths = []

        if self.use_depth_inp:
            if self.only_loc_feats:
                pred_depths = traj_preds[range(len(traj_preds)), output_inds, 4]
            else:
                pred_depths = traj_preds[range(len(traj_preds)), output_inds, 8]
            pred_depths = pred_depths.split(list(num_instances))
        if inst_scores is not None:
            full_score_result = []
        for b_ind, (b_mask_preds, bboxes, orig_class, feats) in enumerate(
            zip(mask_preds, pred_bboxes, orig_classes, feat_preds)
        ):
            if inst_scores is not None:
                score_result = []
                full_score_result.append(score_result)
            #scene_seg = torch.zeros(1024, 2048, device=input_feats.device)
            scene_seg = starting_seg[b_ind]
            if self.use_depth_sorting:
                seq_depths = pred_depths[b_ind]
                inst_depths, inst_order = seq_depths.sort(descending=True)
                if background_depths is not None:
                    current_depths = background_depths[b_ind]
            else:
                inst_order = range(len(bboxes))
            scene_results = []
            scene_classes = []
            scene_logit_classes = []
            scene_logits = []
            scene_bboxes = []
            scene_feats = []
            scene_unscaled_masks = []
            scene_depths = []
            final_result.append(scene_results)
            final_classes.append(scene_classes)
            final_logits.append(scene_logits)
            final_logit_classes.append(scene_logit_classes)
            final_logit_bboxes.append(scene_bboxes)
            final_feats.append(scene_feats)
            final_depths.append(scene_depths)
            final_unscaled_masks.append(scene_unscaled_masks)
            for id,inst_ind in enumerate(inst_order):
                mask_pred, bbox = b_mask_preds[inst_ind:inst_ind+1], bboxes[inst_ind:inst_ind+1]
                pasted_mask = model_utils.paste_mask(mask_pred.unsqueeze(1), bbox, 1024, 2048,
                                                      self.use_bbox_ulbr)
                pasted_logits = pasted_mask.clamp(0.01, 0.99)
                pasted_logits = torch.log(pasted_logits/(1-pasted_logits))
                scene_logits.append(pasted_logits.squeeze(0))
                scene_logit_classes.append(orig_class[inst_ind]+11)
                scene_bboxes.append(bbox.squeeze(0))
                scene_feats.append(feats[inst_ind])
                scene_depths.append(seq_depths[inst_ind])
                scene_unscaled_masks.append(mask_pred)
                pasted_mask = (pasted_mask >= 0.5).long()*(id+1)*1000
                pasted_mask = pasted_mask.squeeze(0)
                if self.use_depth_sorting and background_depths is not None:
                    raise NotImplementedError()
                else:
                    scene_seg = (pasted_mask > 0)*pasted_mask + \
                                (~(pasted_mask > 0))*scene_seg
            for id, inst_ind in enumerate(inst_order):
                current_seg = (scene_seg == (id+1)*1000).long()
                if current_seg.sum() > 0:
                    scene_results.append(current_seg)
                    scene_classes.append(orig_class[inst_ind]+11)
                    if inst_scores is not None:
                        score_result.append(inst_scores[b_ind][inst_ind].item())
        result_dict = {
            'instances': final_result,
            'instance_classes': final_classes,
            'logits': final_logits,
            'logit_classes': final_logit_classes,
            'logit_bboxes': final_logit_bboxes,
            'feats': final_feats,
            'depths': final_depths,
            'unscaled_masks': final_unscaled_masks,
        }
        if inst_scores is not None:
            result_dict['instance_scores'] = full_score_result
        return result_dict

