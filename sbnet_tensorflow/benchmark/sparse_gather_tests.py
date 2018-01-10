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

import numpy as np
import tensorflow as tf

from sparse_conv_lib import convert_mask_to_block_indices, convert_mask_to_indices_custom
from sparse_conv_lib import calc_block_params
from sparse_conv_lib import _calc_block_strides, _pad_input
from sparse_conv_lib import sbnet_module


def gather_tf(x, mask, bsize, ksize, strides, padding):
    blk_indices = convert_mask_to_block_indices(mask, bsize, ksize, strides, padding, 0.0)
    blk_shape = tf.shape(blk_indices)
    blk_indices_ = tf.reshape(blk_indices, [-1, 3])

    # Calculate the block strides.
    bstrides = _calc_block_strides(blk_shape, ksize, strides)

    # Pad input.
    x_ = _pad_input(
        x, ksize, strides, padding, bsize=[1, blk_shape[1], blk_shape[2], 1], bstrides=bstrides)

    # Gather patches.
    p = tf.gather_nd(x_, blk_indices_)

    # Reshape patches.
    p = tf.reshape(p, [blk_shape[0], blk_shape[1], blk_shape[2], -1])
    return p, blk_indices


def gather_custom(x, mask, bsize, ksize, strides, padding):
    x_shape = [int(ss) for ss in x.get_shape()]
    block_params = calc_block_params(x_shape, bsize, ksize, strides, padding)
    indices = convert_mask_to_indices_custom(mask, block_params, 0.0)
    p = sbnet_module.sparse_gather(
        x,
        indices.bin_counts,
        indices.active_block_indices,
        bsize=block_params.bsize,
        boffset=block_params.boffset,
        bstride=block_params.bstrides)
    return p, indices


class SparseGatherTests(tf.test.TestCase):
    def _test_sparse_gather(self, mask, x, w, bsize, ksize, strides, padding):
        with tf.Session() as sess:
            mask = tf.constant(mask)
            x = tf.constant(x)
            w = tf.constant(w)
            a_tf, _ = gather_tf(x, mask, bsize, ksize, strides, padding)
            a_custom, ind = gather_custom(x, mask, bsize, ksize, strides, padding)

            a1, a2, active, num = sess.run(
                [a_tf, a_custom, ind.active_block_indices, ind.bin_counts])
            num = num[0]
            sortIdx = active[:num].argsort()
            a2 = a2[sortIdx]
            np.testing.assert_array_equal(a1, a2)

    def test_basic(self):
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
        # x = np.ones([1, mask.shape[1], mask.shape[2], 1], dtype=np.float32)
        x = np.arange(mask.shape[1] * mask.shape[2]).reshape([1, mask.shape[1], mask.shape[2],
                                                              1]).astype(np.float32)
        w = np.ones(ksize, dtype=np.float32)
        self._test_sparse_gather(mask, x, w, bsize, ksize, strides, padding)

    def test_large(self):
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
        # x = np.ones([1, mask.shape[1], mask.shape[2], 1], dtype=np.float32)
        x = np.arange(mask.shape[1] * mask.shape[2]).reshape([1, mask.shape[1], mask.shape[2],
                                                              1]).astype(np.float32)
        w = np.ones(ksize, dtype=np.float32)
        self._test_sparse_gather(mask, x, w, bsize, ksize, strides, padding)


if __name__ == '__main__':
    tf.test.main()
