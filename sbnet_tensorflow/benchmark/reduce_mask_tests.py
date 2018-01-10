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

from sparse_conv_lib import convert_mask_to_indices, convert_mask_to_indices_custom
from sparse_conv_lib import calc_block_params


class ReduceMaskTests(tf.test.TestCase):
    def _test_reduce_mask(self, mask, bsize, ksize, strides, padding):
        with tf.Session():
            mask = tf.constant(mask)
            indices = convert_mask_to_indices(mask, bsize, ksize, strides, padding, 0.0)
            x_shape = [1] + [int(ss) for ss in mask.get_shape()[1:]] + [1]
            block_params = calc_block_params(x_shape, bsize, ksize, strides, padding)
            indices_custom = convert_mask_to_indices_custom(mask, block_params, 0.0)

            activeBlockIndicesResult = indices_custom.active_block_indices.eval()
            binCountsResult = indices_custom.bin_counts.eval()
            activeBlockIndicesResult = activeBlockIndicesResult[:binCountsResult[0]]
            sortIdx = activeBlockIndicesResult.argsort()
            activeBlockIndicesResult = activeBlockIndicesResult[sortIdx]
            clippedResults = np.copy(activeBlockIndicesResult.view(np.uint16))
            clippedResults = clippedResults.reshape([-1, 4])[:, [2, 1, 0]]
            indices_val = indices.eval()
            np.testing.assert_array_equal(indices_val, clippedResults)

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
        self._test_reduce_mask(mask, bsize, ksize, strides, padding)

    def test_larger(self):
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
        self._test_reduce_mask(mask, bsize, ksize, strides, padding)


if __name__ == '__main__':
    tf.test.main()
