# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
from torch import nn




class DistWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels):
        return self.model.loss(inputs, labels)

    def save(self, path):
        return self.model.save(path)

    def load(self, path):
        return self.model.load(path)