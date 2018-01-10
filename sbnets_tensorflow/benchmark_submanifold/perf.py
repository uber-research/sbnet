"""

   Sparse Blocks Network
   Copyright (c) 2017, Uber Technologies, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""


from __future__ import division, print_function

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import torchnet
import torch.nn as nn
import numpy as np
import sparseconvnet.legacy as scn
import time
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import io
import requests
import os
import six
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../benchmark/"))
from cu_prof import cu_prof_start, cu_prof_stop


def submanifold_single_conv(inchan, outchan):
    dtype = 'torch.cuda.FloatTensor'
    model = scn.Sequential()
    model.add(scn.ValidConvolution(2, inchan, outchan, 3, False))
    model.type(dtype)
    model.cuda()
    return model


def regular_single_conv(inout_chan, bottleneck_chan):
    dtype = 'torch.cuda.FloatTensor'
    model = nn.Sequential(nn.Conv2d(inout_chan, bottleneck_chan, 3, padding=1, bias=False))
    model.type(dtype)
    model.cuda()
    return model


def submanifold_resnet_block(inout_chan, bottleneck_chan):
    # relu is fused in this implementation
    dtype = 'torch.cuda.FloatTensor'

    bn1 = scn.BatchNormReLU(inout_chan)
    bn2 = scn.BatchNormReLU(bottleneck_chan)
    bn3 = scn.BatchNormReLU(bottleneck_chan)
    bn1.train = False
    bn2.train = False
    bn3.train = False
    model = scn.Sequential() \
        .add(bn1) \
        .add(scn.ValidConvolution(2, inout_chan, bottleneck_chan, 3, False)) \
        .add(bn2) \
        .add(scn.ValidConvolution(2, bottleneck_chan, bottleneck_chan, 3, False)) \
        .add(bn3) \
        .add(scn.ValidConvolution(2, bottleneck_chan, inout_chan, 3, False))
    model.type(dtype)
    model.cuda()

    return model


def regular_resnet_block(inout_chan, bottleneck_chan):
    dtype = 'torch.cuda.FloatTensor'
    model = nn.Sequential( \
        nn.BatchNorm2d(inout_chan), \
        nn.ReLU(), \
        nn.Conv2d(inout_chan, bottleneck_chan, 3, padding=1, bias=False), \
        nn.BatchNorm2d(bottleneck_chan), \
        nn.ReLU(), \
        nn.Conv2d(bottleneck_chan, bottleneck_chan, 3, padding=1, bias=False), \
        nn.BatchNorm2d(bottleneck_chan), \
        nn.ReLU(), \
        nn.Conv2d(bottleneck_chan, inout_chan, 3, padding=1, bias=False))
    model.type(dtype)
    model.cuda()
    return model


def generate_top_left_mask(xsize, sparsity):
    # xsize is NCHW
    density = 1.0 - sparsity
    edge_ratio = np.sqrt(density)
    height = int(np.ceil(edge_ratio * xsize[2]))
    width = int(np.ceil(edge_ratio * xsize[3]))
    x = np.zeros(xsize, dtype=np.float32)
    x[:, :, :height, :width] = 1.0
    return x


def run_dense(img_tensor, out_chan, res_block=False, n_warmup=15, n_run=15):
    stream = torch.cuda.current_stream()
    img_tensor_cu = img_tensor.cuda()
    input_size = img_tensor_cu.shape
    in_chan = int(input_size[1])
    if not res_block:
        denseModel = regular_single_conv(in_chan, out_chan)
    else:
        denseModel = regular_resnet_block(in_chan, out_chan)

    # warmup
    var = Variable(img_tensor_cu)
    for i in six.moves.xrange(n_warmup):
        out = denseModel.forward(var)

    cu_prof_start()
    starte = torch.cuda.Event(enable_timing=True)
    ende = torch.cuda.Event(enable_timing=True)
    stream.record_event(starte)
    for i in six.moves.xrange(n_run):
        out = denseModel.forward(var)
    stream.record_event(ende)
    ende.synchronize()
    evt_dt = starte.elapsed_time(ende)
    cu_prof_stop()
    outDense = out
    dense_ms = evt_dt / n_run
    return dense_ms


def run_sparse(img_tensor, out_chan, res_block=False, n_warmup=15, n_run=15):
    stream = torch.cuda.current_stream()
    input_size = img_tensor.shape
    batch = input_size[0]
    in_chan = input_size[1]
    h = input_size[2]
    w = input_size[3]
    if not res_block:
        sparseModel = submanifold_single_conv(in_chan, out_chan)
    else:
        sparseModel = submanifold_resnet_block(in_chan, out_chan)
    sparse_batch = scn.InputBatch(2, torch.LongTensor([h, w]))
    sparse_batch.addSample()
    count = 0
    for y, x in np.ndindex((h, w)):
        val = img_tensor[0, :, y, x]
        if val[0] > 0.5:
            location = torch.LongTensor([y, x])
            featureVector = torch.FloatTensor(val)
            sparse_batch.setLocation(location, featureVector, 0)
            count += 1

    # warmup
    inp = sparse_batch.cuda()
    for i in range(n_warmup):
        out = sparseModel.forward(inp)

    cu_prof_start()

    # Use CUDA events for accurate timing.
    starte = torch.cuda.Event(enable_timing=True)
    ende = torch.cuda.Event(enable_timing=True)
    stream.record_event(starte)
    for i in range(n_run):
        out = sparseModel.forward(inp)
    stream.record_event(ende)
    ende.synchronize()
    evt_dt = starte.elapsed_time(ende)
    cu_prof_stop()
    sparse_ms = evt_dt / n_run
    return sparse_ms
