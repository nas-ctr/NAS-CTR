#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-09-16

import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.utils import NASCTR, get_logger

BLOCKS = {
    'FM':
        lambda params: FM('FM', params),
    'MLP-32':
        lambda params: MLP('MLP-32', 32, params),
    'MLP-64':
        lambda params: MLP('MLP-64', 64, params),
    'MLP-128':
        lambda params: MLP('MLP-128', 128, params),
    'MLP-256':
        lambda params: MLP('MLP-256', 256, params),
    'MLP-512':
        lambda params: MLP('MLP-512', 512, params),
    'MLP-1024':
        lambda params: MLP('MLP-1024', 1024, params),
    'ElementWise-sum':
        lambda params: ElementWise('ElementWise-sum', 'sum', params),
    'ElementWise-avg':
        lambda params: ElementWise('ElementWise-avg', 'avg', params),
    'ElementWise-min':
        lambda params: ElementWise('ElementWise-min', 'min', params),
    'ElementWise-max':
        lambda params: ElementWise('ElementWise-max', 'max', params),
    'ElementWise-innerproduct':
        lambda params: ElementWise('ElementWise-innerproduct', 'innerproduct', params),
    'Crossnet-1':
        lambda params: CrossNet('Crossnet-1', 1, params),
    'Crossnet-2':
        lambda params: CrossNet('Crossnet-2', 2, params),
    'Crossnet-3':
        lambda params: CrossNet('Crossnet-3', 3, params),
    'Crossnet-4':
        lambda params: CrossNet('Crossnet-4', 4, params),
    'Crossnet-5':
        lambda params: CrossNet('Crossnet-5', 5, params),
    'Crossnet-6':
        lambda params: CrossNet('Crossnet-6', 6, params),
}

LOGGER = get_logger(NASCTR)


class Block(nn.Module):
    """
    The input shape of raw sparse feature is (batch_size, field_size, embedding_dim).
    The input shape of raw dense feature is (batch_size, field_size, embedding_dim).
    The input shape of inner block is (batch_size, features_size).
    """

    def __init__(self, block_name, params, use_batchnorm=True, use_relu=True, use_dropout=True, dropout_rate=0.5, use_linear=True):
        super(Block, self).__init__()
        self._block_name = block_name
        self._block_in_dim = params['block_in_dim']
        self._block_out_dim = params['block_out_dim']
        self._embedding_dim = params['embedding_dim']
        self._num_sparse_feats = params['num_sparse_feats']
        self._num_dense_feats = params['num_dense_feats']
        self._num_feats = params['num_feats']
        self._num_inputs = params['num_inputs']
        if use_linear:
            if self._num_dense_feats > 0:
                self._raw_dense_linear = nn.Linear(params['raw_dense_linear'], self._block_out_dim)
            if self._num_sparse_feats > 0:
                self._raw_sparse_linear = nn.Linear(params['raw_sparse_linear'], self._block_out_dim)

        self._use_batchnorm = use_batchnorm
        self._use_relu = use_relu
        self._use_dropout = use_dropout
        self._dropout_rate = dropout_rate

        self._relu = nn.ReLU()
        self._batchnorm = nn.BatchNorm1d(self._block_out_dim)
        self._dropout = nn.Dropout(self._dropout_rate)

    def forward(self, inputs):
        """
        :param inputs: list, e.g. [(x1, input_type1), (x2, input_type2)]
        input_type == 0 means empty
        input_type == 1 means raw dense features
        input_type == 2 means raw sparse features
        input_type == 3 means inner block output features
        """
        raise NotImplementedError

    @property
    def name(self):
        return self._block_name

    @property
    def num_params(self):
        return self._num_params

    def _count_params(self):
        return sum([p.numel() for p in self.parameters() if p is not None and p.requires_grad])


class FM(Block):
    """
    This block applies FM. The 2-D array will be converted into 3-D array.
    """

    def __init__(self, block_name, params, use_batchnorm=True, use_relu=True, use_dropout=True, dropout_rate=0.5):
        super(FM, self).__init__(block_name, params, use_batchnorm=use_batchnorm, use_relu=use_relu,
                                 use_dropout=use_dropout, dropout_rate=dropout_rate, use_linear=False)
        self._emb_linear = nn.Linear(self._block_in_dim, self._embedding_dim)
        self._output_linear = nn.Linear(self._embedding_dim, self._block_out_dim)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        x_empty = None
        for x, input_type in inputs:
            if input_type == 0:
                x_empty = x
                continue
            if x is None:
                raise Exception(f"input type {input_type}, but got NONE input")
            if input_type == 3:
                x = self._emb_linear(x)
            if len(x.shape) == 2:
                x = torch.unsqueeze(x, dim=1)
            if x_empty is not None:
                x += x_empty
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        sum_squared = torch.pow(torch.sum(x, dim=1), 2)
        squared_sum = torch.sum(torch.pow(x, 2), dim=1)
        second_order = torch.sub(sum_squared, squared_sum)
        final = 0.5 * second_order
        output = self._output_linear(final)
        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        return output


class MLP(Block):
    """
    This block applies MLP. The 3-D array will be converted into 3-D array.
    """

    def __init__(self, block_name, hidden_size, params, use_batchnorm=True, use_relu=True, use_dropout=True,
                 dropout_rate=0.5):
        super(MLP, self).__init__(block_name, params, use_batchnorm=use_batchnorm, use_relu=use_relu,
                                  use_dropout=use_dropout, dropout_rate=dropout_rate)
        self._hidden_size = hidden_size
        self._hidden_linear = nn.Linear(self._block_in_dim * self._num_inputs, hidden_size)
        self._output_linear = nn.Linear(hidden_size, self._block_out_dim)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        x_empty = None
        for x, input_type in inputs:
            if input_type == 0:
                x_empty = x
                continue
            if x is None:
                raise Exception(f"input type {input_type}, but got NONE input")
            if len(x.shape) == 3:
                x = torch.reshape(x, (x.shape[0], -1))
            if input_type == 1:
                x = self._raw_dense_linear(x)
            elif input_type == 2:
                x = self._raw_sparse_linear(x)
            if x_empty is not None:
                x += x_empty
            x_list.append(x)
        x = torch.cat(x_list, dim=1)

        final = self._hidden_linear(x)
        output = self._output_linear(final)

        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        return output


class ElementWise(Block):
    """
    This block applies inner product. The 3-D array will be converted into 3-D array.
    The elementwise type should be avg, sum, min, max or innerproduct.
    """

    def __init__(self, block_name, elementwise_type, params, use_batchnorm=True, use_relu=True, use_dropout=True,
                 dropout_rate=0.5):
        super(ElementWise, self).__init__(block_name, params,
                                          use_batchnorm=use_batchnorm, use_relu=use_relu, use_dropout=use_dropout,
                                          dropout_rate=dropout_rate)
        self._elementwise_type = elementwise_type

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        x_empty = None
        for x, input_type in inputs:
            if input_type == 0:
                x_empty = x
                continue
            if x is None:
                raise Exception(f"input type {input_type}, but got NONE input")
            if len(x.shape) == 3:
                x = torch.reshape(x, (x.shape[0], -1))
            if input_type == 1:
                x = self._raw_dense_linear(x)
            elif input_type == 2:
                x = self._raw_sparse_linear(x)
            if x_empty is not None:
                x += x_empty
            x_list.append(x)
        if len(x_list) == 1:
            return x_list[0]
        else:
            x = torch.stack(x_list, dim=0)

        if self._elementwise_type == 'avg':
            final = torch.mean(x, dim=0)
        elif self._elementwise_type == 'sum':
            final = torch.sum(x, dim=0)
        elif self._elementwise_type == 'min':
            final, _ = torch.min(x, dim=0)
        elif self._elementwise_type == 'max':
            final, _ = torch.max(x, dim=0)
        elif self._elementwise_type == 'innerproduct':
            final = torch.prod(x, dim=0)
        else:
            final = torch.sum(x, dim=0)

        output = final

        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        return output


class CrossNet(Block):
    """
    This block applies CrossNet. The 3-D array will be converted into 3-D array.
    """

    def __init__(self, block_name, layer_num, params, use_batchnorm=True, use_relu=True, use_dropout=True,
                 dropout_rate=0.5):
        super(CrossNet, self).__init__(block_name, params,
                                       use_batchnorm=use_batchnorm, use_relu=use_relu, use_dropout=use_dropout,
                                       dropout_rate=dropout_rate)
        self._layer_num = layer_num
        self._w = nn.Parameter(torch.randn(self._layer_num, self._block_in_dim * self._num_inputs))
        nn.init.xavier_uniform_(self._w)
        self._b = nn.Parameter(torch.randn(self._layer_num, self._block_in_dim * self._num_inputs))
        nn.init.zeros_(self._b)
        self._bn_list = nn.ModuleList()
        for _ in range(self._layer_num + 1):
            self._bn_list.append(nn.BatchNorm1d(self._block_in_dim * self._num_inputs))
        self._output_linear = nn.Linear(self._block_in_dim * self._num_inputs, self._block_out_dim)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x_list = []
        x_empty = None
        for x, input_type in inputs:
            if input_type == 0:
                x_empty = x
                continue
            if x is None:
                raise Exception(f"input type {input_type}, but got NONE input")
            if len(x.shape) == 3:
                x = torch.reshape(x, (x.shape[0], -1))
            if input_type == 1:
                x = self._raw_dense_linear(x)
            elif input_type == 2:
                x = self._raw_sparse_linear(x)
            if x_empty is not None:
                x += x_empty
            x_list.append(x)
        x = torch.cat(x_list, dim=1)

        x = self._bn_list[0](x)
        x0 = x
        for i in range(self._layer_num):
            w = torch.unsqueeze(self._w[i, :].T, dim=1)                  # In * 1
            xw = torch.mm(x, w)                                       # None * 1
            x = torch.mul(x0, xw) + self._b[i, :] + x   # None * In
            x = self._bn_list[i + 1](x)

        output = self._output_linear(x)
        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        return output
