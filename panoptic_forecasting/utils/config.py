# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import argparse
import yaml
import os
import copy

def convert_val(val):
    if val[0] == '[' and val[-1] == ']':
        val = [convert_val(x.strip()) for x in val[1:-1].split(',')]
    elif val in ['True', 'true']:
        val = True
    elif val in ['False', 'false']:
        val = False
    else:
        try:
            i_val = int(val)
        except:
            i_val = None
        try:
            f_val = float(val)
        except:
            f_val = None
        if i_val is not None and '.' not in val:
            val = i_val
        elif f_val is not None:
            val = f_val
    return val

def load_config(extra_args=None):
    parser = argparse.ArgumentParser('')
    parser.add_argument('--working_dir', required=True)
    parser.add_argument('--config_file')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--load_model')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--load_best_model', action='store_true')
    parser.add_argument('--extra_args', nargs=2, action='append')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    if extra_args is not None:
        for extra_arg in extra_args:
            parser.add_argument(extra_arg[0], **extra_arg[1])

    args = parser.parse_args()

    if args.load_best_model or args.continue_training:
        config_path = os.path.join(args.working_dir, 'config.yaml')
        with open(config_path, 'r') as fin:
            params = yaml.safe_load(fin)
    elif args.load_model:
        config_path = os.path.join(os.path.dirname(args.load_model), 'config.yaml')
        with open(config_path, 'r') as fin:
            params = yaml.safe_load(fin)
    else:
        params = {}

    config_path = args.config_file
    if config_path is not None:
        with open(config_path, 'r') as fin:
            new_params = yaml.safe_load(fin)
        params = merge_config(params, new_params)
    params = merge_config(params, vars(args))
    if args.extra_args is not None:
        for name, val in args.extra_args:
            name = name.split('.')
            val = convert_val(val)
            tmp_params = params
            for n_part in name[:-1]:
                if n_part not in tmp_params:
                    tmp_params[n_part] = {}
                tmp_params = tmp_params[n_part]
            tmp_params[name[-1]] = val
    return params


def merge_config(old, new):
    result = {}
    keys = set(old.keys()).union(set(new.keys()))
    for key in keys:
        if key in old and not key in new:
            result[key] = old[key]
        elif key not in old and key in new:
            result[key] = new[key]
        elif isinstance(old[key], dict):
            result[key] = merge_config(old[key], new[key])
        else:
            result[key] = new[key]
    return result



