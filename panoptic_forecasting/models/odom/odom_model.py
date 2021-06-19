# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn

from panoptic_forecasting.models.base_model import BaseModel

class OdomModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        self.predict_type = params['model']['predict_type']
        self.normalize_input = params['model'].get('normalize_input')
        if self.normalize_input:
            odom_norm_params = params['data'].get('odom_norm_params')
            if odom_norm_params is None:
                mean = torch.zeros(2)
                std = torch.zeros(2)
            else:
                mean, std = odom_norm_params
            self.odom_mean = nn.Parameter(mean.unsqueeze(0), requires_grad=False)
            self.odom_std = nn.Parameter(std.unsqueeze(0), requires_grad=False)
        if self.predict_type not in ['direct', 'offset']:
            raise ValueError('predict_type not recognized: ',self.predict_type)
        inp_emb_layers = params['model'].get('inp_emb_layers')
        if inp_emb_layers is not None:
            inp_emb_layers.insert(0, 2)
            layers = []
            for s1, s2 in zip(inp_emb_layers[:-1], inp_emb_layers[1:]):
                layers.append(nn.Linear(s1, s2))
                layers.append(nn.ReLU())
            self.inp_emb = nn.Sequential(*layers)
            inp_size = inp_emb_layers[-1]
        else:
            self.inp_emb = None
            inp_size = 2

        rnn_hidden = params['model'].get('rnn_hidden')
        self.rnn = nn.GRU(inp_size, rnn_hidden, batch_first=True)
        out_layers = params['model'].get('out_layers', [])
        out_layers.insert(0, rnn_hidden)
        out_layers.append(2)
        layers = []
        for idx, (s1, s2) in enumerate(
                zip(out_layers[:-1], out_layers[1:])):
            if idx > 0:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(s1, s2))
        self.out = nn.Sequential(*layers)
        loss_type = params['model']['loss_fn']
        self.use_normalized_loss = params['model'].get('use_normalized_loss')
        if loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError('loss_fn not recognized:', self.loss_fn)

    def _normalize(self, inp):
        old_shape = inp.shape
        inp = inp.reshape(-1, old_shape[-1])
        normalized_inp = (inp - self.odom_mean) / self.odom_std
        return normalized_inp.reshape(old_shape)

    def _unnormalize(self, inp):
        old_shape = inp.shape
        inp = inp.reshape(-1, old_shape[-1])
        result = inp * self.odom_std + self.odom_mean
        return result.reshape(old_shape)

    def forward(self, inps, output_len):
        if self.normalize_input:
            inps = self._normalize(inps)
        if self.inp_emb is not None:
            rnn_inps = self.inp_emb(inps)
        else:
            rnn_inps = inps
        out, hidden = self.rnn(rnn_inps[:, :-1])
        current_val = inps[:, -1].unsqueeze(1)
        results = []
        for _ in range(output_len):
            if self.inp_emb is not None:
                current_inp = self.inp_emb(current_val)
            else:
                current_inp = current_val
            out, hidden = self.rnn(current_inp, hidden)
            out = self.out(out)
            if self.predict_type == 'offset':
                current_val = current_val + out
            else:
                current_val = out
            results.append(current_val)
        results = torch.cat(results, dim=1)
        if self.normalize_input:
            normalized_results = results
            results = self._unnormalize(results)
        else:
            normalized_results = self._normalize(results)
        return results, normalized_results

    def loss(self, inputs, labels):
        inp_odom = inputs['odometry']
        label_odom = labels['odometry']

        preds, normalized_preds = self(inp_odom, label_odom.size(1))
        if self.use_normalized_loss:
            normalized_label = self._normalize(label_odom)
            loss = self.loss_fn(normalized_preds, normalized_label)
        else:
            loss = self.loss_fn(preds, label_odom)
        loss = loss.reshape(loss.size(0), -1).mean(1)
        return {'loss': loss}

    def predict(self, inputs, labels):
        inp_odom = inputs['odometry']
        label_odom = labels['odometry']
        preds, _ = self(inp_odom, label_odom.size(1))
        return {'odometry': preds}


