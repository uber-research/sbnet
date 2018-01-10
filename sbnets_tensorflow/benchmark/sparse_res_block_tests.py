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
# Unit tests for sparse_conv.py
#
from __future__ import division, print_function, unicode_literals

import itertools
import numpy as np
import os
import tensorflow as tf

from collections import namedtuple

from sparse_conv_lib import _get_offset_array
from sparse_conv_lib import calc_block_params
from sparse_conv_lib import convert_mask_to_block_indices
from sparse_conv_lib import convert_mask_to_indices_custom
from sparse_conv_lib import mask_conv2d
from sparse_conv_lib import sparse_conv2d
from sparse_conv_lib import sparse_conv2d_custom
from sparse_conv_lib import sparse_conv2d_matmul
from sparse_conv_lib import upsample_indices
from sparse_conv_lib import calc_block_params_res_block
from sparse_conv_lib import sparse_res_block_bottleneck
from sparse_conv_lib import res_block_bottleneck
from tensorflow.python.ops.gradient_checker import compute_gradient
from tensorflow.python.ops import gradient_checker

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def cosine_angle(v1, v2):
    # eps = np.finfo(np.float32).eps
    v1_norm = np.sqrt(np.dot(v1, v1))
    v2_norm = np.sqrt(np.dot(v2, v2))
    if v1_norm == 0.0 and v2_norm == 0.0:
        return 1.0, 1.0, 1.0
    if v1_norm == 0.0:
        v1_norm = 1.0
    if v2_norm == 0.0:
        v2_norm = 1.0
    cosine = np.dot(v1, v2) / v1_norm / v2_norm
    cosine = min(max(cosine, -1.0), 1.0)
    return cosine, v1_norm, v2_norm


def get_degree(radian):
    return radian * 180 / np.pi


def compute_gradient_angle(x,
                           x_shape,
                           y,
                           y_shape,
                           x_init_value=None,
                           delta=1e-3,
                           init_targets=None,
                           extra_feed_dict=None):
    grad = compute_gradient(
        x, x_shape, y, y_shape, x_init_value, delta, init_targets, extra_feed_dict=extra_feed_dict)
    if isinstance(grad, tuple):
        grad = [grad]
    error = 0
    for j_t, j_n in grad:
        if j_t.size or j_n.size:    # Handle zero size tensors correctly
            #error = np.maximum(error, np.fabs(j_t - j_n).max())
            # print(j_t.shape, j_n.shape)
            # xxx'
            # print('grad1', j_t.ravel())
            # print('grad2', j_n.ravel())
            cosine, norm1, norm2 = cosine_angle(j_t.ravel(), j_n.ravel())
            # print('Jt', j_t.ravel()[:10], 'norm', norm1)
            # print('Jn', j_n.ravel()[:10], 'norm', norm2)
            error = get_degree(np.arccos(cosine))
    return error


def compute_gradient_abs_error(x,
                               x_shape,
                               y,
                               y_shape,
                               x_init_value=None,
                               delta=1e-3,
                               init_targets=None,
                               extra_feed_dict=None):
    grad = compute_gradient(
        x, x_shape, y, y_shape, x_init_value, delta, init_targets, extra_feed_dict=extra_feed_dict)
    if isinstance(grad, tuple):
        grad = [grad]
    error = 0
    atol = 1e-5
    for j_t, j_n in grad:
        if j_t.size or j_n.size:    # Handle zero size tensors correctly
            idx = np.logical_and(j_t - j_n > atol, np.fabs(j_t) > atol)
            # error = np.maximum(error, np.fabs((j_t[idx] - j_n[idx]) / j_t[idx]).max())
            error = np.maximum(error, np.fabs(j_t - j_n).max())
    return error


class ResBlockGradientTests(tf.test.TestCase):
    def _test_resblock_gradients(self, xval, maskval, bsize, strides, padding, data_format='NHWC'):
        with tf.Graph().as_default() as g:
            x = tf.constant(xval)
            mask = tf.constant(maskval)
            ch_in = xval.shape[3]
            ch_out = xval.shape[3] // 4
            ksize_list = [[1, 1, ch_in, ch_out], [3, 3, ch_out, ch_out], [1, 1, ch_out, ch_in]]
            y = res_block_bottleneck(
                x,
                ksize_list,
                strides,
                is_training=True,
                data_format=data_format,
                w_project=None,
                no_activation=False)
            trainable_vars = tf.trainable_variables()
            print('')
            print('-' * 55)
            print('Dense Residual')
            print('{:30s} {:>10s} {:>10s}'.format('name', 'grad angle', 'abs err'))
            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                yval = y.eval()
                err = compute_gradient_angle(x, xval.shape, y, yval.shape, x_init_value=xval)
                err2 = compute_gradient_abs_error(x, xval.shape, y, yval.shape, x_init_value=xval)
                print('{:30s} {:>10.3f} {:>10.3f}'.format('x', err, err2))

                for name in [
                        'sub3/conv3/Conv2D:0', 'sub3/relu3:0', 'sub3/bn3/FusedBatchNorm:0',
                        'sub2/conv2/Conv2D:0', 'sub2/relu2:0', 'sub2/bn2/FusedBatchNorm:0',
                        'sub1/conv1/Conv2D:0', 'sub1/relu1:0', 'sub1/bn1/FusedBatchNorm:0'
                ]:
                    act = g.get_tensor_by_name(name)
                    actval = act.eval()
                    err = compute_gradient_angle(
                        act, actval.shape, y, yval.shape, x_init_value=actval)
                    err2 = compute_gradient_abs_error(
                        act, actval.shape, y, yval.shape, x_init_value=actval)
                    print('{:30s} {:>10.3f} {:>10.3f}'.format(name, err, err2))

                # self.assertTrue(err < 0.001)
                for vv in trainable_vars:
                    vvval = vv.eval()
                    err = compute_gradient_angle(vv, vvval.shape, y, yval.shape, x_init_value=vvval)
                    err2 = compute_gradient_abs_error(
                        vv, vvval.shape, y, yval.shape, x_init_value=vvval)
                    print('{:30s} {:>10.3f} {:>10.3f}'.format(vv.name, err, err2))
                    # self.assertTrue(err < 0.001)

    def test_resblock_gradients(self):
        bsize = [1, 5, 5, 1]
        ksize = [3, 3, 4, 4]
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        rnd = np.random.RandomState(0)
        mask = rnd.uniform(-1, 1, [3, 9, 9]).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)
        xval = rnd.uniform(-1, 1, [mask.shape[0], mask.shape[1], mask.shape[2],
                                   ksize[2]]).astype(np.float32)
        self._test_resblock_gradients(xval, mask, bsize, strides, padding, data_format='NHWC')


class SparseResBlockGradientTests(tf.test.TestCase):
    def _test_sparse_resblock_gradients(self,
                                        xval,
                                        maskval,
                                        bsize,
                                        strides,
                                        padding,
                                        data_format='NHWC'):
        with tf.Graph().as_default() as g:
            x = tf.constant(xval)
            mask = tf.constant(maskval)
            ch_in = xval.shape[3]
            ch_out = xval.shape[3] // 4
            ksize_list = [[1, 1, ch_in, ch_out], [3, 3, ch_out, ch_out], [1, 1, ch_out, ch_in]]
            blk_params = calc_block_params_res_block(xval.shape, bsize, ksize_list, strides,
                                                     padding)
            ind = convert_mask_to_indices_custom(mask, blk_params, 0.)
            ReduceMask = namedtuple('ReduceMask', ['active_block_indices', 'bin_counts'])
            ind.active_block_indices.set_shape([27])
            ind.bin_counts.set_shape([1])
            ind_var = tf.Variable(ind.active_block_indices, trainable=False)
            bin_var = tf.Variable(ind.bin_counts, trainable=False)
            ind_fixed = ReduceMask(active_block_indices=ind_var, bin_counts=bin_var)
            tf_ind = convert_mask_to_indices_custom(mask, blk_params, 0.)
            with self.test_session() as sess:
                py_inds = sess.run([tf_ind])
            ind = lambda: 0
            ind.bin_counts = tf.constant(py_inds[0].bin_counts)
            ind.active_block_indices = tf.constant(py_inds[0].active_block_indices)

            y = sparse_res_block_bottleneck(
                x,
                ksize_list,
                ind_fixed,
                blk_params,
                strides,
                is_training=True,
                data_format=data_format,
                w_project=None,
                no_activation=False,
                use_var=False)
            trainable_vars = tf.trainable_variables()
            print('')
            print('-' * 55)
            print('Sparse Residual')
            print('{:30s} {:>10s} {:>10s}'.format('name', 'grad angle', 'abs err'))
            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                yval = y.eval()
                err = compute_gradient_angle(x, xval.shape, y, yval.shape, x_init_value=xval)
                err2 = compute_gradient_abs_error(x, xval.shape, y, yval.shape, x_init_value=xval)
                print('{:30s} {:>10.3f} {:>10.3f}'.format('x', err, err2))

                #'sub3/bn3/batchnorm/add_1:0',
                for name in [
                        'SparseScatter:0', 'SparseGather:0', 'sub3/bn3/FusedBatchNorm:0',
                        'sub3/conv3/Conv2D:0', 'sub3/relu3:0', 'sub2/conv2/Conv2D:0',
                        'sub2/relu2:0', 'sub2/bn2/FusedBatchNorm:0', 'sub1/conv1/Conv2D:0',
                        'sub1/relu1:0', 'sub1/bn1/FusedBatchNorm:0'
                ]:
                    act = g.get_tensor_by_name(name)
                    actval = act.eval()
                    err = compute_gradient_angle(
                        act, actval.shape, y, yval.shape, x_init_value=actval)
                    err2 = compute_gradient_abs_error(
                        act, actval.shape, y, yval.shape, x_init_value=actval)
                    print('{:30s} {:>10.3f} {:>10.3f}'.format(name, err, err2))

                for vv in trainable_vars:
                    vvval = vv.eval()
                    err = compute_gradient_angle(vv, vvval.shape, y, yval.shape, x_init_value=vvval)
                    err2 = compute_gradient_abs_error(
                        vv, vvval.shape, y, yval.shape, x_init_value=vvval)
                    print('{:30s} {:>10.3f} {:>10.3f}'.format(vv.name, err, err2))

    def test_sparse_resblock_gradients(self):
        bsize = [1, 5, 5, 1]
        ksize = [3, 3, 4, 4]
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        rnd = np.random.RandomState(0)
        mask = rnd.uniform(0, 1, [3, 9, 9]).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)
        xval = rnd.uniform(-0.1, 0.1, [mask.shape[0], mask.shape[1], mask.shape[2], ksize[2]
                                       ]).astype(np.float32) + 1.0
        self._test_sparse_resblock_gradients(
            xval, mask, bsize, strides, padding, data_format='NHWC')


class SparseConv2DGradientTests(tf.test.TestCase):
    def _test_sparse_conv2d_gradient(self, mask, bsize, ksize, strides, padding, transpose=False):
        # Currently we don't care about VALID convolution.
        assert padding == 'SAME', 'We do not support VALID conv at the moment.'
        use_var = False
        mask_ = tf.constant(mask)
        blk_params = calc_block_params(
            list(mask.shape) + [ksize[2]], bsize, ksize, strides, padding)
        ind = convert_mask_to_indices_custom(mask_, blk_params, 0.)
        ReduceMask = namedtuple('ReduceMask', ['active_block_indices', 'bin_counts'])
        ind.active_block_indices.set_shape([27])
        ind.bin_counts.set_shape([1])
        ind_var = tf.Variable(ind.active_block_indices, trainable=False)
        bin_var = tf.Variable(ind.bin_counts, trainable=False)
        ind_fixed = ReduceMask(active_block_indices=ind_var, bin_counts=bin_var)
        rnd = np.random.RandomState(0)
        batch_size = 1
        xval = rnd.uniform(-0.1, 0.1, [mask.shape[0], mask.shape[1], mask.shape[2],
                                       ksize[2]]).astype(np.float32)
        x = tf.constant(xval)
        wval = rnd.uniform(-1, 1, ksize).astype(np.float32)
        w = tf.constant(wval)
        y = sparse_conv2d_custom(
            x, w, ind_fixed, blk_params, strides, use_var=use_var, transpose=transpose)
        print('')
        print('-' * 55)
        print('Sparse Conv Layer')
        print('{:30s} {:>10s} {:>10s}'.format('name', 'grad angle', 'abs err'))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            yval = y.eval()
            err = compute_gradient_angle(x, xval.shape, y, yval.shape, x_init_value=xval)
            err2 = compute_gradient_abs_error(x, xval.shape, y, yval.shape, x_init_value=xval)
            print('{:30s} {:>10.3f} {:>10.3f}'.format('x', err, err2))

            err = compute_gradient_angle(w, wval.shape, y, yval.shape, x_init_value=wval)
            err = compute_gradient_abs_error(w, wval.shape, y, yval.shape, x_init_value=wval)
            print('{:30s} {:>10.3f} {:>10.3f}'.format('w', err, err2))

    def test_sparse_conv2d_gradient(self):
        bsize = [1, 5, 5, 1]
        ksize = [3, 3, 4, 4]
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        rnd = np.random.RandomState(0)
        mask = rnd.uniform(0, 1, [3, 9, 9]).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)
        self._test_sparse_conv2d_gradient(mask, bsize, ksize, strides, padding, transpose=False)


if __name__ == '__main__':
    tf.test.main()
