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

from sparse_conv_lib import sbnet_module
from sparse_conv_lib import calc_block_params, convert_mask_to_indices_custom
from tf_conv_dims import calc_out_size_4d_np
from sparse_gather_tests import gather_tf


def scatter_tf(q, blk_indices, out_shape):
    q_shape = tf.shape(q)
    blk_indices_crop = blk_indices[:, :q_shape[1], :q_shape[2], :]
    y = tf.scatter_nd(blk_indices_crop, q, out_shape)
    return y


def scatter_custom(q, indices, out_shape, bsize_out, boffset, bstride):
    y = tf.zeros(out_shape, dtype=q.dtype)
    y = sbnet_module.sparse_scatter(
        q,
        indices.bin_counts,
        indices.active_block_indices,
        y,
        bsize=bsize_out,
        boffset=boffset,
        bstride=bstride,
        add=False)
    return y


class SparseScatterTests(tf.test.TestCase):
    def _test_sparse_scatter(self, mask, x, w, out_shape, bsize, ksize, strides, padding):
        with tf.Session() as sess:
            x = tf.constant(x)
            w = tf.constant(w)
            p, blk_indices = gather_tf(x, mask, bsize, ksize, strides, padding)
            block_params = calc_block_params([int(ss) for ss in x.get_shape()], bsize, ksize,
                                             strides, padding)
            ind_custom = convert_mask_to_indices_custom(mask, block_params, 0.0)
            p_custom = sbnet_module.sparse_gather(
                x,
                ind_custom.bin_counts,
                ind_custom.active_block_indices,
                bsize=block_params.bsize,
                bstride=block_params.bstrides,
                boffset=block_params.boffset)
            p_shape = [
                int(x.get_shape()[0]), block_params.bsize[0], block_params.bsize[1],
                int(x.get_shape()[3])
            ]
            q = tf.nn.conv2d(p, w, strides, 'VALID')
            q_custom = tf.nn.conv2d(p_custom, w, strides, 'VALID')
            y_tf = scatter_tf(q, blk_indices, out_shape)
            q_shape = calc_out_size_4d_np(p_shape, ksize, strides, 'VALID')
            bsize_out = [q_shape[1], q_shape[2]]
            boffset = [0, 0]
            y_custom = scatter_custom(q_custom, ind_custom, out_shape, bsize_out, boffset,
                                      block_params.bstrides)
            p1, p2, q_val, y1, y2, active, num = sess.run([
                p, p_custom, q, y_tf, y_custom, ind_custom.active_block_indices,
                ind_custom.bin_counts
            ])
            num = num[0]
            sortIdx = active[:num].argsort()
            p2 = p2[sortIdx]

            # Make sure p's are the same.
            np.testing.assert_array_equal(p1, p2)

            # Check y's are the same.
            np.testing.assert_array_equal(y1, y2)

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
        mask = tf.constant(mask)
        w = np.ones(ksize, dtype=np.float32)
        out_shape = [1, 5, 5, 1]
        self._test_sparse_scatter(mask, x, w, out_shape, bsize, ksize, strides, padding)


if __name__ == '__main__':
    tf.test.main()
