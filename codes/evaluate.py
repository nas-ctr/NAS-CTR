#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-09-30
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def evaluate(model, data_queue, args):
    model.eval()
    xs, ys = [], []
    with torch.no_grad():
        tt = tqdm(data_queue, smoothing=0, mininterval=1.0)
        for raw_dense, raw_sparse, labels in tt:
            if args.use_gpu:
                raw_dense = raw_dense.cuda(non_blocking=True)
                raw_sparse = raw_sparse.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            predicts, regs = model((raw_dense, raw_sparse))
            xs.append(predicts.flatten())
            ys.append(labels)
        predicts, labels = torch.cat(xs), torch.cat(ys)

        loss = model.compute_loss(predicts, labels, regs, use_reg=False, use_arch_loss=False)
        predicts = torch.sigmoid(predicts)
        auc = roc_auc_score(labels.cpu(), predicts.cpu())
    return loss.cpu().detach().item(), auc
