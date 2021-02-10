#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-09-12
import os
import sys
import traceback

import argparse
import logging
import time
import torch
import torch.backends.cudnn as cudnn

from codes.evaluate import evaluate
from codes.network import DynamicNetwork
from codes.train import train_search
from codes.utils import NASCTR, create_exp_dir, count_parameters_in_mb, get_logger, PROJECT_PATH, check_directory, \
    set_seed, MyDistributedDataParallel, format_arch_p, EarlyStop

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()

parser.add_argument('--train_ratio', type=float, default=0.75)
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=16384)
parser.add_argument('--use_pretrained_embeddings', type=bool, default=True)
parser.add_argument('--arch_lr', type=float, default=0.001)
parser.add_argument('--arch_weight_decay', type=float, default=1e-6)
parser.add_argument('--weights_reg', type=float, default=1e-6)
parser.add_argument('--opt', type=str, default='SGD', choices=['Adam', 'AdamW', 'SGD', 'Adagrad'])
# for Adam
parser.add_argument('--adam_lr', type=float, default=0.001)
parser.add_argument('--adam_weight_decay', type=float, default=1e-6)
# for AdamW
parser.add_argument('--adamw_lr', type=float, default=0.001)
parser.add_argument('--adamw_weight_decay', type=float, default=1e-6)
# for SGD
parser.add_argument('--sgd_lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--sgd_lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--sgd_momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--sgd_weight_decay', type=float, default=3e-4)
# for Adagrad
parser.add_argument('--adagrad_lr', type=float, default=0.001)
parser.add_argument('--adagrad_weight_decay', type=float, default=3e-4)

parser.add_argument('--arch_reg', type=float, default=0, choices=[0, 0.05, 0.5, 1, 2])
parser.add_argument('--grad_clip', type=float, default=10, help='gradient clipping')
parser.add_argument('--unrolled', type=bool, default=False, help='whether to use second gradient')
parser.add_argument('--e_greedy', type=float, default=0)
parser.add_argument('--search_epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--block_in_dim', type=int, default=400)
parser.add_argument('--block_out_dim', type=int, default=400)
parser.add_argument('--embedding_dim', type=int, default=16)
parser.add_argument('--num_block', type=int, default=7)
parser.add_argument('--num_free_block', type=int, default=2)
parser.add_argument('--max_skip_connect', type=int, default=10)
parser.add_argument('--block_keys', type=list, default=None)
parser.add_argument('--dataset', type=str, default='criteo')
parser.add_argument('--dataset_path', type=str, default=None)
parser.add_argument('--mode', type=str, default='nasctr', help='choose how to search')
parser.add_argument('--model_name', type=str, default='nasctr')

parser.add_argument('--gpu', type=int, default=3, help="gpu divice id")
parser.add_argument('--use_gpu', type=bool, default=False)
parser.add_argument('--seed', type=int, default=666, help="random seed")
parser.add_argument('--log_save_dir', type=str, default=PROJECT_PATH)
parser.add_argument('--show_log', type=bool, default=False)


args = parser.parse_args()
set_seed(args.seed)
timestamp = time.strftime("%Y%m%d-%H%M%S")
MODEL_PATH = f'experiments/{args.dataset}/{timestamp}'
create_exp_dir(MODEL_PATH, scripts_to_save=None, force_removed=True)
LOG_ROOT_PATH = os.path.join(PROJECT_PATH, 'logs')
check_directory(LOG_ROOT_PATH, True)
log_save_dir = os.path.join(LOG_ROOT_PATH, f'log_search_{args.dataset}_{args.model_name}_{args.num_block}_{args.embedding_dim}_{args.block_in_dim}_{args.batch_size}_{args.opt}_{args.seed}.txt')
LOGGER = get_logger(NASCTR, log_save_dir, level=logging.DEBUG)


def main(dataset, args):
    if args.use_gpu and not torch.cuda.is_available():
        LOGGER.error('no gpu device available')
        sys.exit(1)
    if args.use_gpu:
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.set_device(args.gpu)
        LOGGER.info('gpu device = %d' % args.gpu)

    LOGGER.info("args = %s", args)
    train_queue, valid_queue, test_queue = dataset.get_search_dataloader(args.train_ratio, args.valid_ratio, args.batch_size)

    LOGGER.info(f"Start searching architecture...")
    search_start = time.time()
    criterion = torch.nn.BCEWithLogitsLoss()
    if args.use_gpu:
        criterion = criterion.cuda()
        model = DynamicNetwork(dataset, args, criterion, logger=LOGGER).cuda()
    else:
        model = DynamicNetwork(dataset, args, criterion, logger=LOGGER)

    LOGGER.info("Total param size = %fMB", count_parameters_in_mb(model))

    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.sgd_lr,
            momentum=args.sgd_momentum,
            weight_decay=args.sgd_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.search_epochs), eta_min=args.sgd_lr_min)
    elif args.opt == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), args.adagrad_lr)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.adam_lr, weight_decay=args.adam_weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), args.adamw_lr, weight_decay=args.adamw_weight_decay)
    arch_optimizer = torch.optim.Adam(model.arch_parameters(), args.arch_lr, weight_decay=args.arch_weight_decay)

    best_auc, best_loss, best_arch = 0, float('inf'), None
    early_stopper = EarlyStop(k=args.patience)
    for i in range(args.search_epochs):
        if args.opt == 'SGD':
            scheduler.step()
            LOGGER.info(f"search_epoch: {i}; lr: {scheduler.get_last_lr()}")
        train_loss, val_loss = train_search(train_queue, valid_queue, model, optimizer, arch_optimizer, args)
        model.binarize()
        test_loss, test_auc = evaluate(model, test_queue, args)
        arch, arch_p, num_params = model.get_arch()
        model.restore()

        LOGGER.info(
            f'search_epoch: {i}, train_loss: {train_loss:.5f},'
            f' val_loss: {val_loss:.5f},'
            f' test_loss: {test_loss:.5f}, test_auc: {test_auc:.5f},'
            f' time spent: {(time.time() - search_start):.5f},'
            f' arch: {arch}, num_params: {num_params}')
        # LOGGER.debug(f"arch_p {arch} \n {format_arch_p(arch_p)}")
        if test_auc > best_auc or (test_auc == best_auc and test_loss < best_loss):
            best_auc = test_auc
            best_loss = test_loss
            best_arch = arch
        is_stop = early_stopper.add_metric(test_auc)
        if is_stop:
            LOGGER.info(f"Not rise for {early_stopper.not_rise_times}, stop search")

    LOGGER.info(f"Finish searching archs, best auc: {best_auc}, best loss: {best_loss}, best_arch: {best_arch}")


if __name__ == '__main__':
    from codes.datasets.avazu import AvazuDataset
    from codes.datasets.criteo import CriteoDataset

    if args.dataset == 'criteo':
        from codes.datasets.criteo import CriteoDataset
        dataset = CriteoDataset(data_path=args.dataset_path, repreprocess=False,
                                use_pretrained_embeddings=args.use_pretrained_embeddings, logger=LOGGER)
    elif args.dataset == 'avazu':
        from codes.datasets.avazu import AvazuDataset
        dataset = AvazuDataset(data_path=args.dataset_path, repreprocess=False,
                               use_pretrained_embeddings=args.use_pretrained_embeddings, logger=LOGGER)
    try:
        main(dataset, args)
    except Exception as e:
        LOGGER.error(traceback.format_exc())
