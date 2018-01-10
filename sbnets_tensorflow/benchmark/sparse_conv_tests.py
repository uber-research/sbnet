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
import tensorflow as tf

from sparse_conv_lib import _get_offset_array
from sparse_conv_lib import calc_block_params
from sparse_conv_lib import convert_mask_to_block_indices
from sparse_conv_lib import convert_mask_to_indices_custom
from sparse_conv_lib import mask_conv2d
from sparse_conv_lib import sparse_conv2d
from sparse_conv_lib import sparse_conv2d_custom
from sparse_conv_lib import sparse_conv2d_matmul
from sparse_conv_lib import upsample_indices


class UpsampleIndicesTests(tf.test.TestCase):
    def test_offset_array(self):
        offset_exp = np.array(
            [
                [
                    [-1, -1],    # YAPF_NO_FORMAT
                    [-1, 0],
                    [-1, 1]
                ],
                [
                    [0, -1],    # YAPF_NO_FORMAT
                    [0, 0],
                    [0, 1]
                ],
                [
                    [1, -1],    # YAPF_NO_FORMAT
                    [1, 0],
                    [1, 1]
                ]
            ],
            dtype=np.int32)
        offset = _get_offset_array([3, 3])
        with self.test_session():
            offset_act = offset.eval()
            np.testing.assert_array_equal(offset_act.shape, [3, 3, 2])
            np.testing.assert_array_equal(offset_act, offset_exp)

    def test_upsample_indices(self):
        ind = np.array(
            [
                [0, 1, 2],    # YAPF_NO_FORMAT
                [0, 2, 0]
            ],
            dtype=np.int32)
        ind = tf.constant(ind)
        ind_up_exp = np.array(
            [
                [0, 1, 2],    # YAPF_NO_FORMAT
                [0, 1, 3],
                [0, 1, 4],
                [0, 2, 2],
                [0, 2, 3],
                [0, 2, 4],
                [0, 3, 2],
                [0, 3, 3],
                [0, 3, 4],
                [0, 2, 0],
                [0, 2, 1],
                [0, 2, 2],
                [0, 3, 0],
                [0, 3, 1],
                [0, 3, 2],
                [0, 4, 0],
                [0, 4, 1],
                [0, 4, 2],
            ],
            dtype=np.int32)
        ind_up = upsample_indices(ind, [1, 3, 3, 1], [1, 1, 1, 1])
        with self.test_session():
            ind_up_act = ind_up.eval()
            np.testing.assert_array_equal(ind_up_act.shape, [2, 3, 3, 3])
            np.testing.assert_array_equal(ind_up_act.reshape([-1, 3]), ind_up_exp)


class SparseConv2DTests(tf.test.TestCase):
    def _test_sparse_conv2d(self, ind_blk, padding, y_exp):
        ind_blk = tf.reshape(ind_blk, [2, 3, 3, 3])
        x = tf.constant(np.ones([1, 5, 5, 1], dtype=np.float32))
        w = tf.constant(np.ones([3, 3, 1, 1], dtype=np.float32))
        y = sparse_conv2d(x, w, ind_blk, [1, 1, 1, 1], padding)
        with self.test_session():
            y_act = y.eval()
            np.testing.assert_array_equal(y_act.reshape(y_exp.shape), y_exp)

    def test_sparse_conv2d_valid(self):
        ind_blk = tf.constant(
            np.array(
                [
                    [0, 1, 2],    # YAPF_NO_FORMAT
                    [0, 1, 3],
                    [0, 1, 4],
                    [0, 2, 2],
                    [0, 2, 3],
                    [0, 2, 4],
                    [0, 3, 2],
                    [0, 3, 3],
                    [0, 3, 4],
                    [0, 2, 0],
                    [0, 2, 1],
                    [0, 2, 2],
                    [0, 3, 0],
                    [0, 3, 1],
                    [0, 3, 2],
                    [0, 4, 0],
                    [0, 4, 1],
                    [0, 4, 2],
                ],
                dtype=np.int32))
        ind_blk = tf.reshape(ind_blk, [2, 3, 3, 3])
        y_exp = np.array(
            [[
                [[0], [0], [0]],    # YPAF_NO_FORAMT
                [[0], [0], [9]],
                [[9], [0], [0]]
            ]],
            dtype=np.float32)
        padding = 'VALID'
        self._test_sparse_conv2d(ind_blk, padding, y_exp)

    def test_sparse_conv2d_same(self):
        ind_blk = tf.constant(
            np.array(
                [
                    [0, 1, 2],    # YAPF_NO_FORMAT
                    [0, 1, 3],
                    [0, 1, 4],
                    [0, 2, 2],
                    [0, 2, 3],
                    [0, 2, 4],
                    [0, 3, 2],
                    [0, 3, 3],
                    [0, 3, 4],
                    [0, 2, 0],
                    [0, 2, 1],
                    [0, 2, 2],
                    [0, 3, 0],
                    [0, 3, 1],
                    [0, 3, 2],
                    [0, 4, 0],
                    [0, 4, 1],
                    [0, 4, 2],
                ],
                dtype=np.int32))
        ind_blk = tf.reshape(ind_blk, [2, 3, 3, 3])
        y_exp = np.array(
            [[
                [[0], [0], [0], [0], [0]],    # YAPF_NO_FORAMT
                [[0], [0], [9], [0], [0]],
                [[6], [0], [0], [0], [0]],
                [[0], [0], [0], [0], [0]],
                [[0], [0], [0], [0], [0]]
            ]],
            dtype=np.float32)
        padding = 'SAME'
        self._test_sparse_conv2d(ind_blk, padding, y_exp)

    def _test_sparse_conv2d_with_mask(self, mask, bsize, ksize, strides, padding, y_exp):
        mask_ = tf.constant(mask)
        ind_blk = convert_mask_to_block_indices(mask_, bsize, ksize, strides, padding, 0.)
        x = tf.constant(np.ones([1, mask.shape[1], mask.shape[2], 1], dtype=np.float32))
        w = tf.constant(np.ones(ksize, dtype=np.float32))
        y = sparse_conv2d(x, w, ind_blk, strides, padding)
        with self.test_session():
            y_act = y.eval()
            self.assertEqual(y_act.size, y_exp.size)
            np.testing.assert_array_equal(y_act.reshape(y_exp.shape), y_exp)

    def test_sparse_conv2d_with_mask_valid(self):
        bsize = [1, 3, 3, 1]
        ksize = [3, 3, 1, 1]
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        mask = np.array(
            [[
                [0, 0, 0, 0, 0],    # YAPF_NO_FORMAT
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]],
            dtype=np.float32)
        y_exp = np.array(
            [[
                [[9], [9], [9]],    # YAPF_NO_FORMAT
                [[9], [9], [9]],
                [[9], [0], [0]],
            ]],
            dtype=np.float32)
        self._test_sparse_conv2d_with_mask(mask, bsize, ksize, strides, padding, y_exp)

    def test_sparse_conv2d_with_mask_same(self):
        bsize = [1, 3, 3, 1]
        ksize = [3, 3, 1, 1]
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        mask = np.array(
            [[
                [0, 0, 0, 0, 0],    # YAPF_NO_FORMAT
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]],
            dtype=np.float32)
        y_exp = np.array(
            [[
                [[0], [6], [6], [6], [0]],    # YAPF_NO_FORMAT
                [[6], [9], [9], [9], [0]],
                [[6], [9], [9], [9], [0]],
                [[6], [9], [0], [0], [0]],
                [[0], [0], [0], [0], [0]]
            ]],
            dtype=np.float32)
        self._test_sparse_conv2d_with_mask(mask, bsize, ksize, strides, padding, y_exp)

    def test_sparse_conv2d_with_mask_same_even_block(self):
        bsize = [1, 4, 4, 1]
        ksize = [3, 3, 1, 1]
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        mask = np.array(
            [[
                [0, 0, 0, 0, 0],    # YAPF_NO_FORMAT
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]],
            dtype=np.float32)
        y_exp = np.array(
            [[
                [[4], [6], [6], [6], [0]],    # YAPF_NO_FORMAT
                [[6], [9], [9], [9], [0]],
                [[6], [9], [9], [9], [0]],
                [[6], [9], [9], [9], [0]],
                [[0], [0], [0], [0], [0]]
            ]],
            dtype=np.float32)
        self._test_sparse_conv2d_with_mask(mask, bsize, ksize, strides, padding, y_exp)

    def test_sparse_conv2d_with_mask_same_even_block_strides(self):
        bsize = [1, 5, 5, 1]
        ksize = [3, 3, 1, 1]
        strides = [1, 2, 2, 1]
        padding = 'SAME'
        mask = np.array(
            [[
                [0, 0, 0, 0, 0],    # YAPF_NO_FORMAT
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]],
            dtype=np.float32)
        y_exp = np.array(
            [[
                [[4], [6], [0]],    # YAPF_NO_FORMAT
                [[6], [9], [0]],
                [[0], [0], [0]]
            ]],
            dtype=np.float32)
        self._test_sparse_conv2d_with_mask(mask, bsize, ksize, strides, padding, y_exp)

    def test_sparse_conv2d_with_large_block_strides(self):
        bsize = [1, 5, 5, 1]
        ksize = [2, 2, 1, 1]
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        mask = np.array(
            [[
                [0, 0, 0, 0, 0, 1, 1, 1],    # YAPF_NO_FORMAT
                [0, 0, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]],
            dtype=np.float32)
        y_exp = np.array(
            [[
                [[4], [4], [4], [4], [4], [4], [4]],    # YAPF_NO_FORMAT
                [[4], [4], [4], [4], [4], [4], [4]],
                [[4], [4], [4], [4], [4], [4], [4]],
                [[4], [4], [4], [4], [4], [4], [4]],
                [[0], [0], [0], [0], [4], [4], [4]],
                [[0], [0], [0], [0], [4], [4], [4]],
                [[0], [0], [0], [0], [4], [4], [4]]
            ]],
            dtype=np.float32)
        self._test_sparse_conv2d_with_mask(mask, bsize, ksize, strides, padding, y_exp)

    def _test_sparse_conv2d_correctness(self,
                                        xsize,
                                        bsize,
                                        ksize,
                                        strides,
                                        padding,
                                        dtype=tf.float32):
        np.random.RandomState(0)
        x = tf.constant(np.random.uniform(-1, 1, xsize), dtype=dtype)
        w = tf.constant(np.random.uniform(-1, 1, ksize), dtype=dtype)
        mask = np.random.uniform(-5, 1, xsize[:-1])
        mask = tf.constant((mask > 0).astype(np.float32), dtype=dtype)
        mask_out = tf.nn.max_pool(
            tf.expand_dims(mask, 3), [1, ksize[0], ksize[1], 1], strides, padding)
        y_exp = mask_conv2d(x, w, mask, strides, padding)
        ind_blk = convert_mask_to_block_indices(mask, bsize, ksize, strides, padding, 0.)
        y_act = sparse_conv2d(x, w, ind_blk, strides, padding)
        with self.test_session():
            y_exp_val = y_exp.eval()
            y_act_val = y_act.eval()
            mask_val = mask_out.eval()
            mask_val = np.tile(mask_val, [1, 1, 1, ksize[-1]])
            y_exp_val = y_exp_val * mask_val
            y_act_val = y_act_val * mask_val
            y_exp_val = y_exp_val[mask_val > 0.]
            y_act_val = y_act_val[mask_val > 0.]
            np.testing.assert_allclose(y_act_val, y_exp_val, rtol=1e-2, atol=1e-5)

    def test_sparse_conv2d_correctness(self):
        xsize = [10, 28, 28, 10]
        padding = 'SAME'
        test_func = self._test_sparse_conv2d_correctness
        for kk in [1, 3, 5]:
            for padding in ['SAME', 'VALID']:
                for ss in [1, 2]:
                    _ksize = [kk, kk, xsize[-1], xsize[-1]]
                    _bsize = [1, kk + 2, kk + 2, 1]
                    _strides = [1, ss, ss, 1]
                    test_func(xsize, _bsize, _ksize, _strides, padding)

    def _test_sparse_conv2d_matmul_correctness(self, xsize, ksize, padding, dtype=tf.float32):
        np.random.RandomState(0)
        strides = [1, 1, 1, 1]    # Only [1, 1, 1, 1] is supported currently.
        # Use block size to be the same with kernel size makes it the same with matrix multiplication.
        bsize = [1, ksize[0], ksize[1], 1]
        x = tf.constant(np.random.uniform(-1, 1, xsize), dtype=dtype)
        w = tf.constant(np.random.uniform(-1, 1, ksize), dtype=dtype)
        mask = np.random.uniform(-5, 1, xsize[:-1])
        mask = tf.constant((mask > 0).astype(np.float32), dtype=dtype)
        mask_out = tf.nn.max_pool(
            tf.expand_dims(mask, 3), [1, ksize[0], ksize[1], 1], strides, padding)
        y_exp = mask_conv2d(x, w, mask, strides, padding)
        ind_blk = convert_mask_to_block_indices(mask, bsize, ksize, strides, padding, 0.)
        y_act = sparse_conv2d_matmul(x, w, ind_blk, strides, padding)
        with self.test_session():
            y_exp_val = y_exp.eval()
            y_act_val = y_act.eval()
            mask_val = mask_out.eval()
            mask_val = np.tile(mask_val, [1, 1, 1, ksize[-1]])
            y_exp_val = y_exp_val * mask_val
            y_act_val = y_act_val * mask_val
            y_exp_val = y_exp_val[mask_val > 0.]
            y_act_val = y_act_val[mask_val > 0.]
            np.testing.assert_allclose(y_act_val, y_exp_val, rtol=1e-2, atol=1e-5)

    def test_sparse_conv2d_matmul_correctness(self):
        xsize = [10, 28, 28, 10]
        padding = 'SAME'
        test_func = self._test_sparse_conv2d_matmul_correctness
        for kk in [1, 3, 5]:
            for padding in ['SAME', 'VALID']:
                _ksize = [kk, kk, xsize[-1], xsize[-1]]
                test_func(xsize, _ksize, padding)


class SparseConv2DCustomTests(tf.test.TestCase):
    def _test_sparse_conv2d_custom_with_mask(self,
                                             mask,
                                             bsize,
                                             ksize,
                                             strides,
                                             padding,
                                             y_exp,
                                             use_var=True,
                                             transpose=False):
        # Currently we don't care about VALID convolution.
        assert padding == 'SAME', 'We do not support VALID conv at the moment.'
        mask_ = tf.constant(mask)
        blk_params = calc_block_params(
            list(mask.shape) + [ksize[2]], bsize, ksize, strides, padding)
        ind = convert_mask_to_indices_custom(mask_, blk_params, 0.)
        xval = np.ones([1, mask.shape[1], mask.shape[2], 1], dtype=np.float32)
        x = tf.constant(xval)
        if use_var:
            x = tf.Variable(x)
        w = tf.constant(np.ones(ksize, dtype=np.float32))
        y = sparse_conv2d_custom(
            x, w, ind, blk_params, strides, use_var=use_var, transpose=transpose)
        # Manually paste the input tensor in the expected output.
        y_exp = (
            y_exp == 0).astype(np.float32) * xval[:, :y_exp.shape[1], :y_exp.shape[2], :] + y_exp
        with self.test_session() as sess:
            if use_var:
                sess.run(tf.variables_initializer([x]))
            y_act = y.eval()
            # print('===============')
            # print('Actual')
            # print(y_act.reshape([y_act.shape[1], y_act.shape[2]]))
            # print('Expected')
            # print(y_exp.reshape([y_exp.shape[1], y_exp.shape[2]]))
            # print(y_exp.shape)
            self.assertEqual(y_act.size, y_exp.size)
            np.testing.assert_array_equal(y_act.reshape(y_exp.shape), y_exp)

    def test_sparse_conv2d_with_mask_same(self):
        bsize = [1, 3, 3, 1]
        ksize = [3, 3, 1, 1]
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        mask = np.array(
            [[
                [0, 0, 0, 0, 0],    # YAPF_NO_FORMAT
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]],
            dtype=np.float32)
        y_exp = np.array(
            [[
                [[0], [6], [6], [6], [0]],    # YAPF_NO_FORMAT
                [[6], [9], [9], [9], [0]],
                [[6], [9], [9], [9], [0]],
                [[6], [9], [0], [0], [0]],
                [[0], [0], [0], [0], [0]]
            ]],
            dtype=np.float32)
        self._test_sparse_conv2d_custom_with_mask(mask, bsize, ksize, strides, padding, y_exp)


if __name__ == '__main__':
    tf.test.main()
