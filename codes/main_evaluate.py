#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-10-06
from codes.utils import NASCTR, create_exp_dir, count_parameters_in_mb, get_logger, PROJECT_PATH, check_directory, \
    set_seed
from codes.train import train
from codes.network import FixedNetwork
from codes.evaluate import evaluate
import torch.backends.cudnn as cudnn
import torch
import time
import logging
import argparse
import traceback
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--train_epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--use_pretrained_embeddings', type=bool, default=False)
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--weights_reg', type=float, default=1e-6)
parser.add_argument('--opt', type=str, default='Adam', choices=['Adam', 'SGD', 'Adagrad', 'AdamW'])
# for Adam
parser.add_argument('--adam_lr', type=float, default=0.001)
parser.add_argument('--adam_weight_decay', type=float, default=1e-6)
# for AdamW
parser.add_argument('--adamw_lr', type=float, default=0.001)
parser.add_argument('--adamw_weight_decay', type=float, default=1e-6)
# for SGD
parser.add_argument('--sgd_lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--sgd_lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--sgd_weight_decay', type=float, default=3e-4)
parser.add_argument('--sgd_momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--sgd_gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--sgd_decay_period', type=int, default=1, help='epochs between two learning rate decays')
# for Adagrad
parser.add_argument('--adagrad_lr', type=float, default=0.001)

parser.add_argument('--block_in_dim', type=int, default=400)
parser.add_argument('--block_out_dim', type=int, default=400)
parser.add_argument('--embedding_dim', type=int, default=16)
parser.add_argument('--num_block', type=int, default=7)
parser.add_argument('--block_keys', type=list, default=None)
parser.add_argument('--dataset', type=str, default='criteo')
parser.add_argument('--dataset_path', type=str, default=None)
parser.add_argument('--arch', type=str, default=None)
parser.add_argument('--mode', type=str, default='nasctr', help='choose how to search')

parser.add_argument('--gpu', type=int, default=3, help="gpu divice id")
parser.add_argument('--use_gpu', type=bool, default=False)
parser.add_argument('--seed', type=int, default=666, help="random seed")
parser.add_argument('--log_save_dir', type=str, default=PROJECT_PATH)
parser.add_argument('--show_log', type=bool, default=True)


args = parser.parse_args()

set_seed(args.seed)
timestamp = time.strftime("%Y%m%d-%H%M%S")
MODEL_PATH = f'experiments/{args.dataset}/{timestamp}'
create_exp_dir(MODEL_PATH, scripts_to_save=None, force_removed=True)
LOG_ROOT_PATH = os.path.join(PROJECT_PATH, 'logs')
check_directory(LOG_ROOT_PATH, force_removed=True)
log_save_dir = os.path.join(
    LOG_ROOT_PATH,
    f'log_evaluate_{args.dataset}_{args.mode}_{args.num_block}_{args.embedding_dim}_{args.block_in_dim}_{args.batch_size}_{args.opt}_{args.seed}.txt')
LOGGER = get_logger(NASCTR, log_save_dir, level=logging.INFO)


def main(dataset, args, arch):
    train_queue, val_queue = dataset.get_eval_dataloader(
        args.train_ratio, args.valid_ratio, args.batch_size)
    test_queue = dataset.get_test_dataloader(args.batch_size)

    if args.use_gpu and not torch.cuda.is_available():
        LOGGER.info('no gpu device available')
        sys.exit(1)
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        LOGGER.info('gpu device = %d' % args.gpu)

    LOGGER.info("args = %s", args)

    eval_start = time.time()
    criterion = torch.nn.BCEWithLogitsLoss()
    if args.use_gpu:
        criterion = criterion.cuda()
        model = FixedNetwork(
            dataset,
            args,
            criterion,
            arch,
            logger=LOGGER).cuda()
    else:
        model = FixedNetwork(dataset, args, criterion, arch, logger=LOGGER)

    LOGGER.info("param size = %fMB", count_parameters_in_mb(model))

    scheduler = None
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.sgd_lr,
            momentum=args.sgd_momentum,
            weight_decay=args.sgd_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=3)
    elif args.opt == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), args.adagrad_lr)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            args.adam_lr,
            weight_decay=args.adam_weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            args.adamw_lr,
            weight_decay=args.adamw_weight_decay)

    val_auc, val_loss = train(train_queue, val_queue, model, optimizer, args, model_path=MODEL_PATH, patience=args.patience,
                              scheduler=scheduler, logger=LOGGER, show_log=args.show_log)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model.pt')))
    test_loss, test_auc = evaluate(model, test_queue, args)
    LOGGER.info(f'Finish evaluate {arch}; '
                f'val_auc: {val_auc:.5f}, val_loss: {val_loss:.5f}; '
                f'test_auc: {test_auc:.5f}, test_loss: {test_loss:.5f}; '
                f'time spent: {(time.time() - eval_start):.5f}')


if __name__ == '__main__':
    from codes.datasets.avazu import AvazuDataset
    from codes.datasets.criteo import CriteoDataset
    arch = None
    if args.dataset == 'avazu':
        dataset = AvazuDataset(
            data_path=args.dataset_path,
            repreprocess=False,
            use_pretrained_embeddings=args.use_pretrained_embeddings,
            logger=LOGGER)
        from codes.architectures import AvazuTestNetwork
        if not arch:
            arch = AvazuTestNetwork
    elif args.dataset == 'criteo':
        dataset = CriteoDataset(
            data_path=args.dataset_path,
            repreprocess=False,
            use_pretrained_embeddings=args.use_pretrained_embeddings,
            logger=LOGGER)
        from codes.architectures import CretioTestNetwork
        if not arch:
            arch = CretioTestNetwork
    try:
        main(dataset, args, arch)
    except Exception as e:
        LOGGER.error(traceback.format_exc())
