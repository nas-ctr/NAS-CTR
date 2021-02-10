#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-09-15
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
from torch.backends import cudnn

NASCTR = 'NAS-CTR'
PROJECT_PATH = os.path.abspath(os.path.join(__file__, '../..'))


def check_directory(path, force_removed=False):
    if force_removed:
        try:
            shutil.rmtree(path)
        except Exception as e:
            pass

    if not os.path.exists(path):
        os.mkdir(path)


def linecount_wc(file):
    return int(os.popen(f'wc -l {file}').read().split()[0])


def create_exp_dir(path, scripts_to_save=None, force_removed=False):
    if force_removed:
        try:
            shutil.rmtree(path)
        except Exception as e:
            pass

    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def get_logger(name, save_dir=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    if save_dir is None:
        return logger

    log_fmt = '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(stream=sys.stdout, level=level, format=log_fmt, datefmt=date_fmt)

    temp = os.path.split(save_dir)
    debug_save_dir = os.path.join(temp[0], 'debug_' + temp[1])
    fh_debug = logging.FileHandler(debug_save_dir)
    debug_filter = logging.Filter()
    debug_filter.filter = lambda record: record.levelno >= level
    fh_debug.addFilter(debug_filter)
    fh_debug.setFormatter(logging.Formatter(log_fmt, date_fmt))
    logger.addHandler(fh_debug)

    fh_info = logging.FileHandler(save_dir)
    info_filter = logging.Filter()
    info_filter.filter = lambda record: record.levelno >= logging.INFO
    fh_info.addFilter(info_filter)
    fh_info.setFormatter(logging.Formatter(log_fmt, date_fmt))
    logger.addHandler(fh_info)

    return logger


def count_parameters_in_mb(model):
    return np.sum(p.numel() for name, p in model.named_parameters() if "auxiliary" not in name
                  and p is not None and p.requires_grad) / 1e6


def concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_gpu(args):
    if args.use_gpu and not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    if args.use_gpu:
        cudnn.benchmark = True
        cudnn.enabled = True
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print('gpu device = %d' % args.gpu)


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def format_arch_p(arch_p):
    string = ''
    for i, p in enumerate(arch_p):
        string += f"block_type{i}: {' '.join(arch_p[i][0].astype(np.str))} \n"
        for j, pp in enumerate(arch_p[i][1]):
            string += f"input{j}: {' '.join(pp.astype(np.str))} \n"
    return string


class EarlyStop:
    def __init__(self, k=3, method='max'):
        self._metrics = []
        self._k = k
        self._not_rise = 0
        self._cur_max = None
        self._method = 'max'

    @property
    def not_rise_times(self):
        return self._not_rise

    def add_metric(self, m):
        if self._method == 'min':
            m = -m
        self._metrics.append(m)
        if self._cur_max is None:
            self._cur_max = m
        if m >= self._cur_max:
            self._cur_max = m
            self._not_rise = 0
        else:
            self._not_rise += 1

        if self._not_rise >= self._k:
            return True
        else:
            return False
