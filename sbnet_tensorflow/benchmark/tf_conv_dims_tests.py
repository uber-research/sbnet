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
# Unit tests for tf_conv_dims.py
#
from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf


class CalcOutSizeTests(tf.test.TestCase):
    def _test_calc_out_size(self, in_size, ksize, stride, padding):
        from tf_conv_dims import calc_out_size_1d
        x = tf.ones([1, in_size, in_size, 1], dtype=tf.float32)
        w = tf.ones([ksize, ksize, 1, 1], dtype=tf.float32)
        y = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding)
        with self.test_session():
            out_size_exp = tf.shape(y)[1].eval()
            out_size_act = calc_out_size_1d(in_size, ksize, stride, padding).eval()
        self.assertEqual(out_size_act, out_size_exp)

    def test_calc_out_size(self):
        for insize in [6, 7, 10, 11]:
            for ksize in [1, 2, 3, 4, 7]:
                for stride in [1, 2, 3]:
                    for padding in ['SAME', 'VALID']:
                        if ksize <= insize:
                            self._test_calc_out_size(insize, ksize, stride, padding)


class CalcOutSizeDeconvTests(tf.test.TestCase):
    def _test_calc_in_size(self, out_size, ksize, stride, padding):
        from tf_conv_dims import calc_out_size_1d_np
        w = tf.ones([ksize, ksize, 1, 1], dtype=tf.float32)
        in_size = calc_out_size_1d_np(out_size, ksize, 1 / float(stride), padding)
        x = tf.ones([1, in_size, in_size, 1], dtype=tf.float32)
        y = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding)
        with self.test_session():
            out_size_exp = tf.shape(y)[1].eval()
        self.assertEqual(out_size, out_size_exp)

    def test_calc_out_size(self):
        for insize in [6, 7, 10, 11]:
            for ksize in [1, 2, 3, 4, 7]:
                for stride in [1, 2, 3]:
                    # Fractional stride methods only works for SAME.
                    # Instead of calc out_size, maybe possible to have another function to calc
                    # in_size
                    for padding in ['SAME']:
                        if ksize <= insize:
                            self._test_calc_in_size(insize, ksize, stride, padding)


class CalcPaddingTests(tf.test.TestCase):
    def test_calc_padding(self):
        from tf_conv_dims import calc_padding_4d
        x = tf.zeros([1, 5, 6, 1])
        p_exp = np.array([0, 1, 1, 1], dtype=np.int32)
        p = calc_padding_4d(tf.shape(x), [2, 3, 1, 1], [1, 1, 1, 1], 'SAME')
        p = tf.stack(p)
        with self.test_session():
            p_act = p.eval()
            np.testing.assert_array_equal(p_act, p_exp)

    def test_calc_padding_err_ksize_list(self):
        from tf_conv_dims import calc_padding_4d
        x = tf.zeros([1, 5, 6, 1])
        err_raised = False
        try:
            calc_padding_4d(tf.shape(x), [2, 3, 1, 1, 1], [2, 1, 1, 1], 'SAME')
        except AssertionError as e:
            self.assertEqual(e.message, 'Expect `ksize` a list/tuple of length 4.')
            err_raised = True
        self.assertTrue(err_raised)

    def test_calc_padding_err_strides_list(self):
        from tf_conv_dims import calc_padding_4d
        x = tf.zeros([1, 5, 6, 1])
        err_raised = False
        try:
            calc_padding_4d(tf.shape(x), [2, 3, 1, 1], [2, 1, 1, 1], 'SAME')
        except AssertionError as e:
            self.assertEqual(e.message, 'Expect first and last dimension of `strides` = 1.')
            err_raised = True
        self.assertTrue(err_raised)

    def test_calc_padding_err_strides_tensor(self):
        from tf_conv_dims import calc_padding_4d
        x = tf.zeros([1, 5, 6, 1])
        err_raised = False
        p = calc_padding_4d(tf.shape(x), [2, 3, 1, 1], tf.constant(np.array([2, 1, 1, 1])), 'SAME')
        p = tf.stack(p)
        with self.test_session():
            try:
                p.eval()
            except tf.errors.InvalidArgumentError as e:
                self.assertTrue(
                    e.message.startswith(
                        'assertion failed: [Expect first and last dimension of `strides` = 1.]'))
                err_raised = True

        self.assertTrue(err_raised)

    def test_calc_padding_valid(self):
        from tf_conv_dims import calc_padding_4d
        x = tf.zeros([1, 5, 5, 1])
        p_exp = np.array([0, 0, 0, 0], dtype=np.int32)
        p = calc_padding_4d(tf.shape(x), [2, 3, 1, 1], [1, 1, 1, 1], 'VALID')
        p = tf.stack(p)
        with self.test_session():
            p_act = p.eval()
            np.testing.assert_array_equal(p_act, p_exp)

    def test_calc_padding_stride(self):
        from tf_conv_dims import calc_padding_4d
        x = tf.zeros([1, 5, 6, 1])
        p_exp = np.array([0, 1, 0, 1], dtype=np.int32)
        p = calc_padding_4d(tf.shape(x), [2, 3, 1, 1], [1, 2, 2, 1], 'SAME')
        p = tf.stack(p)
        with self.test_session():
            p_act = p.eval()
            np.testing.assert_array_equal(p_act, p_exp)


if __name__ == '__main__':
    tf.test.main()
