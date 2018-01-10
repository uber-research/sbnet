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

#
# Sparse convolution performance profile utilities.
#
from __future__ import print_function

import ctypes
import cv2
import itertools
import numpy as np
import os
import six
import tensorflow as tf
import time

from collections import namedtuple
from sparse_conv_lib import sparse_conv2d, sparse_conv2d_custom
from sparse_conv_lib import convert_mask_to_block_indices, convert_mask_to_indices_custom
from sparse_conv_lib import calc_block_params, calc_block_params_res_block
from sparse_conv_lib import res_block_bottleneck, sparse_res_block_bottleneck
from sparse_conv_lib import cuda_timer_start_op, cuda_timer_end_op
from tensorflow.python.framework.errors_impl import ResourceExhaustedError, InternalError

N_REPEAT = 5
N_WARMUP = 3
N_RUN = 3


def generate_top_left_mask(xsize, sparsity):
    """
    Generates a square top-left mask with a target sparsity value.

    :param xsize:       [list]      List of 4 int.
    :param sparsity:    [float]     Target sparsity value.

    :return:            [Tensor]    A tensor with shape to be `xsize` and contains a square of 1's
                                    and the rest being 0's.
    """
    density = 1.0 - sparsity
    edge_ratio = np.sqrt(density)
    height = int(np.ceil(edge_ratio * xsize[1]))
    width = int(np.ceil(edge_ratio * xsize[2]))
    x = np.zeros(xsize[:-1], dtype=np.float32)
    x[:, :height, :width] = 1.0
    return x


TestConfig = namedtuple(
    'TestConfig', ['xsize', 'ksize', 'bsize', 'strides', 'padding', 'is_sparse', 'tol', 'avgpool'])

TestResult = namedtuple('TestResult', ['avg_time', 'block_sparsity'])

ReduceMask = namedtuple('ReduceMask', ['active_block_indices', 'bin_counts'])

TestGraph = namedtuple('TestGraph', ['x_init', 'mask_init', 'bin_init', 'ind_init', 'y', 'dt'])


def _sparse_res_block_with_mask(x, ksize_list, block_params, strides, ind_init, bin_init):
    """Sparse conv 2d with mask."""
    ind_obj = ReduceMask(active_block_indices=ind_init, bin_counts=bin_init)
    y_ = sparse_res_block_bottleneck(
        x, ksize_list, ind_obj, block_params, strides, True, use_var=True, data_format='NCHW')
    return y_


def _sparse_conv2d_custom_with_mask(x, w, block_params, strides, ind_init, bin_init):
    """Sparse conv 2d with mask."""
    ind_obj = ReduceMask(active_block_indices=ind_init, bin_counts=bin_init)
    y_ = sparse_conv2d_custom(x, w, ind_obj, block_params, strides, use_var=True, transpose=True)
    return y_


def _sparse_conv2d_with_mask(x, w, strides, padding, mask, bsize, tol):
    """Sparse conv 2d with mask."""
    ksize = [int(ss) for ss in w.get_shape()]
    ind_blk = convert_mask_to_block_indices(mask, bsize, ksize, strides, padding, tol, avgpool=True)
    y = sparse_conv2d(x, w, ind_blk, strides, padding)
    return y


# MASK_FOLDER = '/mnt/yyz_data_0/users/byang10/pva_det/output/delete_me/odtac_4d_val_vis/mask/'
# MASK = cv2.imread(MASK_FOLDER + 'odtac_4d_val_vis_1000-0000_label.png')


def run_block_sparsity(sess, mask, config):
    block_params = calc_block_params(config.xsize, config.bsize, config.ksize, config.strides,
                                     config.padding)
    ind = convert_mask_to_indices_custom(mask, block_params, config.tol, config.avgpool)
    ind_val, bin_val = sess.run([ind.active_block_indices, ind.bin_counts])
    block_density = bin_val[0] / float(ind_val.shape[0])
    return 1 - block_density


def _build_res_block(mask, config, x_init, ind_init, bin_init, n_repeat=N_REPEAT):
    """Buildds a computation graph for a single residual block."""
    ksize_list = [[1, 1, config.ksize[2], config.ksize[3]]]
    ksize_list += [[3, 3, config.ksize[3], config.ksize[3]]]
    ksize_list += [[1, 1, config.ksize[3], config.ksize[2]]]
    xs = []
    ys = []
    if config.is_sparse:
        with tf.control_dependencies([mask]):
            dt0 = cuda_timer_start_op("my_timer")
            block_params = calc_block_params_res_block(config.xsize, config.bsize, ksize_list,
                                                       config.strides, config.padding)
            ind = convert_mask_to_indices_custom(mask, block_params, config.tol, config.avgpool)
        for _ in six.moves.xrange(n_repeat):
            x_ = tf.Variable(x_init)
            with tf.control_dependencies(ys + [dt0]):
                with tf.variable_scope('sparse_{}'.format(_)):
                    y_ = _sparse_res_block_with_mask(x_, ksize_list, block_params, config.strides,
                                                     ind_init, bin_init)
                xs.append(x_)
                ys.append(y_)
    else:
        ind = None
        for _ in six.moves.xrange(n_repeat):
            x_ = tf.Variable(tf.transpose(x_init, [0, 3, 1, 2]))    # NCHW
            with tf.control_dependencies([x_]):
                dt0 = cuda_timer_start_op("my_timer")
            with tf.control_dependencies(ys + [dt0]):
                with tf.variable_scope('dense_{}'.format(_)):
                    y_ = res_block_bottleneck(
                        x_,
                        ksize_list,
                        config.strides,
                        True,
                        data_format='NCHW',
                        w_project=None,
                        no_activation=False)
                xs.append(x_)
                ys.append(y_)
    with tf.control_dependencies(ys):
        dt = cuda_timer_end_op("my_timer")
        with tf.control_dependencies([dt]):
            y = tf.no_op()
    return y, ind, dt


def _build_conv(mask, config, x_init, ind_init, bin_init, n_repeat=N_REPEAT):
    """Builds a computation graph for a single convolution."""
    wnp = np.random.uniform(-1, 1, config.ksize)    # filter is RSCK
    w = tf.constant(wnp, dtype=tf.float32)
    # AP: Tensorflow doesn't support KCRS from my investigation
    #wt = tf.constant(np.transpose(wnp, [3, 2, 0, 1]), dtype=tf.float32) # transpose to KCRS
    xs = []
    ys = []
    if config.is_sparse:
        with tf.control_dependencies([mask]):
            dt0 = cuda_timer_start_op("my_timer")
            block_params = calc_block_params(config.xsize, config.bsize, config.ksize,
                                             config.strides, config.padding)
            ind = convert_mask_to_indices_custom(mask, block_params, config.tol, config.avgpool)
        for _ in six.moves.xrange(n_repeat):
            x_ = tf.Variable(x_init)    # no need to transpose here since gather/scatter transpose
            with tf.control_dependencies(ys + [dt0]):
                y_ = _sparse_conv2d_custom_with_mask(x_, w, block_params, config.strides, ind_init,
                                                     bin_init)
                xs.append(x_)
                ys.append(y_)
    else:
        ind = None
        for _ in six.moves.xrange(n_repeat):
            x_ = tf.Variable(tf.transpose(x_init, [0, 3, 1, 2]))    # NCHW
            with tf.control_dependencies([x_]):
                dt0 = cuda_timer_start_op("my_timer")
            with tf.control_dependencies(ys + [dt0]):
                y_ = tf.nn.conv2d(x_, w, config.strides, config.padding, data_format='NCHW')
                xs.append(x_)
                ys.append(y_)
    with tf.control_dependencies(ys):
        dt = cuda_timer_end_op("my_timer")
        with tf.control_dependencies([dt]):
            y = tf.no_op()
    return y, ind, dt


def run_one(sess,
            mask,
            config,
            res_block=False,
            options=None,
            run_metadata=None,
            n_warmup=N_WARMUP,
            n_run=N_RUN,
            n_repeat=N_REPEAT):
    """Runs a single setting timing.

    :param sess:         [object]      TensorFlow Session object.
    :param config:       [object]      TestConfig object.
    :param res_block:    [bool]        Whether do single convolution or residual block.
    :param options:      [object]      Session run options.
    :param run_metadata  [object]      RunMetadata object.
    :param n_warmup      [int]         Number of warm-up runs.
    :param n_run         [int]         Number of runs for timing.

    :return:             [object]      TestResult object.
    """
    # Placeholder is needed when x's size is larger than 2GB.
    x_init = tf.placeholder(tf.float32, config.xsize)
    ind_init = tf.placeholder(tf.int64)
    bin_init = tf.placeholder(tf.int32)
    if config.is_sparse:
        mask = tf.constant(mask)

    # Build computation graph.
    if not res_block:
        y, ind, dt = _build_conv(mask, config, x_init, ind_init, bin_init, n_repeat=n_repeat)
    else:
        y, ind, dt = _build_res_block(mask, config, x_init, ind_init, bin_init, n_repeat=n_repeat)

    # Initialize inputs.
    sess.run(
        tf.global_variables_initializer(),
        feed_dict={x_init: np.random.uniform(-1, 1, config.xsize).astype(np.float32)})

    # Sparse indices.
    if ind is not None:
        ind_val, bin_val = sess.run([ind.active_block_indices, ind.bin_counts])
        block_density = bin_val[0] / float(ind_val.shape[0])
        feed_dict = {ind_init: ind_val, bin_init: bin_val}
    else:
        block_density = 1.0
        feed_dict = None

    # Warm up.
    for ii in six.moves.xrange(n_warmup):
        sess.run(y, feed_dict=feed_dict)

    # Actual timing.
    all_dt = []
    for trial in six.moves.xrange(n_run):
        _, dtval = sess.run(
            [y, dt], options=options, run_metadata=run_metadata, feed_dict=feed_dict)
        all_dt.append(dtval)
    avg_time = np.array(all_dt).mean() / n_repeat

    return TestResult(avg_time=avg_time, block_sparsity=1 - block_density)
