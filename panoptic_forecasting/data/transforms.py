# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import random
import numbers

import torch
import torchvision
from PIL import Image
from PIL import ImageOps
import cv2
# The next import prevents hanging when using multiple dataloader threads
cv2.setNumThreads(0)
import numpy as np


'''
These transforms are adapted from https://github.com/NVIDIA/semantic-segmentation/blob/master/transforms/joint_transforms.py.
License reproduced at bottom of the file
'''

class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """
    def __init__(self, size, ignore_index=0, nopad=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, imgs, masks, arrs, arr_interp_types=None, centroid=None):
        w, h = imgs[0].size
        # ASSUME H, W
        th, tw = self.size
        if w == tw and h == th:
            return imgs, mask, arrs

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                print("BORDER: ",border)
                print("FILL: ",self.pad_color)
                imgs = [ImageOps.expand(img, border=border, fill=self.pad_color)
                        for img in imgs]
                if isinstance(mask, list):
                    mask = [ImageOps.expand(m, border=border, fill=self.ignore_index)
                            for m in mask]
                else:
                    mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                arrs = [np.pad(arr, border, constant_values=-1)
                        for arr in arrs]
                w, h = imgs[0].size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        imgs = [img.crop((x1, y1, x1 + tw, y1 + th))
                for img in imgs]
        if isinstance(masks, list):
            masks = [m.crop((x1, y1, x1+tw, y1+th)) for m in mask]
        else:
            masks = masks.crop((x1, y1, x1 + tw, y1 + th))
        if arrs is not None:
            arrs = [arr[y1:y1+th, x1:x1+tw] for arr in arrs]
        return imgs, masks, arrs

# Randomly resizes input/output masks, then crops
# Adapted from https://github.com/NVIDIA/semantic-segmentation/blob/master/transforms/joint_transforms.py#L346
# to handle case where both input and output are masks
class RandomSizeAndCropMasks(object):
    def __init__(self, size, crop_nopad,
                 scale_min=0.5, scale_max=2.0, ignore_index=0, pre_size=None):
        self.size = size
        self.crop = RandomCrop(self.size,
                               ignore_index=ignore_index,
                               nopad=crop_nopad)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.pre_size = pre_size

    def __call__(self, imgs, mask, arrs, arr_interp_types=None, centroid=None):
        #assert imgs[0].size == mask.size
        if arr_interp_types is None and arrs is not None:
            arr_interp_types = [cv2.INTER_NEAREST for _ in arrs]
        # first, resize such that shorter edge is pre_size
        if self.pre_size is None:
            scale_amt = 1.
        elif imgs[0].size[1] < imgs[0].size[0]:
            scale_amt = self.pre_size / imgs[0].size[1]
        else:
            scale_amt = self.pre_size / imgs[0].size[0]
        scale_amt *= random.uniform(self.scale_min, self.scale_max)
        w, h = [int(i * scale_amt) for i in imgs[0].size]

        if centroid is not None:
            centroid = [int(c * scale_amt) for c in centroid]

        imgs = [img.resize((w, h), Image.NEAREST)
                for img in imgs]
        if isinstance(mask, list):
            mask = [m.resize((w,h), Image.NEAREST) for m in mask]
        else:
            mask = mask.resize((w, h), Image.NEAREST)
        if arrs is not None:
            arr_results = []
            for tmp_idx,(arr, interp) in enumerate(zip(arrs, arr_interp_types)):
                try:
                    if len(arr.shape) == 3 and arr.shape[2] == 0:
                        result_arr = np.empty((h,w, 0))
                    else:
                        result_arr = cv2.resize(arr, dsize=(w,h), interpolation=interp)
                        if len(arr.shape) == 3 and arr.shape[2] == 1:
                            result_arr = result_arr[:, :, None]
                    arr_results.append(result_arr)
                except Exception as e:
                    print("OFFENDING IDX: ",tmp_idx)
                    raise e
            arrs = arr_results
            #arrs = [cv2.resize(arr, dsize=(w,h), interpolation=interp)
            #        for arr, interp in zip(arrs, arr_interp_types)]

        return self.crop(imgs, mask, arrs, centroid)


class RandomSizeAndCropMasks_Faster(object):
    def __init__(self, size, crop_nopad,
                 scale_min=0.5, scale_max=2.0, ignore_index=0, pre_size=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.pre_size = pre_size
        self.nopad = crop_nopad
        self.ignore_index = ignore_index
        self.pad_color = (0, 0, 0)

    def __call__(self, segs, mask, arrs, imgs=None, arr_interp_types=None, centroid=None):
        # First, figure out scaling size
        #assert imgs[0].size == mask.size
        if arr_interp_types is None and arrs is not None:
            arr_interp_types = [cv2.INTER_NEAREST for _ in arrs]
        # first, resize such that shorter edge is pre_size
        if self.pre_size is None:
            scale_amt = 1.
        elif segs[0].size[1] < segs[0].size[0]:
            scale_amt = self.pre_size / segs[0].size[1]
        else:
            scale_amt = self.pre_size / segs[0].size[0]
        scale_amt *= random.uniform(self.scale_min, self.scale_max)
        crop_w, crop_h = [int(i * scale_amt) for i in self.size]

        w, h = segs[0].size
        if crop_h > h:
            pad_h = (crop_h - h) // 2 + 1
        else:
            pad_h = 0
        if crop_w > w:
            pad_w = (crop_w - w) // 2 + 1
        else:
            pad_w = 0
        border = (pad_w, pad_h, pad_w, pad_h)
        # Do padding if necessary
        if pad_h or pad_w:
            #print("BORDER: ",border)
            #print("FILL: ",self.pad_color)
            segs = [ImageOps.expand(img, border=border, fill=self.ignore_index)
                    for img in segs]
            if isinstance(mask, list):
                mask = [ImageOps.expand(m, border=border, fill=self.ignore_index)
                        for m in mask]
            else:
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
            arr_border = [(pad_h, pad_h), (pad_w, pad_w), (0, 0)]
            arrs = [np.pad(arr, arr_border, constant_values=0)
                    for arr in arrs]
            if imgs is not None:
                imgs = [ImageOps.expand(img, border=border, fill=0) for img in imgs]
        w, h = segs[0].size
        if w == crop_w:
            x1 = 0
        else:
            x1 = random.randint(0, w - crop_w)
        if h == crop_h:
            y1 = 0
        else:
            y1 = random.randint(0, h - crop_h)

        segs = [img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
                for img in segs]
        if isinstance(mask, list):
            mask = [m.crop((x1, y1, x1+crop_w, y1+crop_h)) for m in mask]
        else:
            mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        if arrs is not None:
            arrs = [arr[y1:y1+crop_h, x1:x1+crop_w] for arr in arrs]
        if imgs is not None:
            imgs = [img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
                    for img in imgs]

        segs = [img.resize(self.size, Image.NEAREST)
                for img in segs]
        if isinstance(mask, list):
            mask = [m.resize(self.size, Image.NEAREST) for m in mask]
        else:
            mask = mask.resize(self.size, Image.NEAREST)
        if arrs is not None:
            arr_results = []
            for tmp_idx,(arr, interp) in enumerate(zip(arrs, arr_interp_types)):
                try:
                    if len(arr.shape) == 3 and arr.shape[2] == 0:
                        result_arr = np.empty((self.size[1],self.size[0], 0))
                    else:
                        result_arr = cv2.resize(arr, dsize=tuple(self.size), interpolation=interp)
                        if len(arr.shape) == 3 and arr.shape[2] == 1:
                            result_arr = result_arr[:, :, None]
                    arr_results.append(result_arr)
                except Exception as e:
                    print("OFFENDING IDX: ",tmp_idx)
                    raise e
            arrs = arr_results
            #arrs = [cv2.resize(arr, dsize=(w,h), interpolation=interp)
            #        for arr, interp in zip(arrs, arr_interp_types)]
        
        if imgs is None:
            return segs, mask, arrs
        else:
            imgs = [img.resize(self.size, Image.BILINEAR) for img in imgs]
            return segs, mask, arrs, imgs

class RandomHorizontallyFlip(object):
    def __call__(self, segs, mask, arrs, imgs=None, arr_interp_types=None):
        if random.random() < 0.5:
            segs = [img.transpose(Image.FLIP_LEFT_RIGHT)
                    for img in segs]
            if isinstance(mask, list):
                mask = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in mask]
            else:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if arrs is not None:
                arrs = [np.fliplr(arr) for arr in arrs]
            if imgs is not None:
                imgs = [img.transpose(Image.FLIP_LEFT_RIGHT)
                        for img in imgs]
        if imgs is None:
            return segs, mask, arrs
        else:
            return segs, mask, arrs, imgs


class Resize(object):
    """
    Resize image to exact size of crop
    """

    def __init__(self, size):
        if isinstance(size, tuple) or isinstance(size, list):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, imgs, mask, arrs, arr_interp_types=None):
        #assert imgs[0].size == mask.size
        if arr_interp_types is None and arrs is not None:
            arr_interp_types = [cv2.INTER_NEAREST for _ in arrs]
        w, h = imgs[0].size
        if (w == h and w == self.size):
            return imgs, mask, arrs
        imgs = [img.resize(self.size, Image.NEAREST)
                for img in imgs]
        if isinstance(mask, list):
            mask = [m.resize(self.size, Image.NEAREST) for m in mask]
        else:
            mask = mask.resize(self.size, Image.NEAREST)
        if arrs is not None:
            arrs = [cv2.resize(arr, dsize=self.size,
                               interpolation=interp)
                    for arr, interp in zip(arrs, arr_interp_types)]
        return imgs, mask, arrs


# MIT License
#
# Copyright (c) 2017 ZijunDeng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.