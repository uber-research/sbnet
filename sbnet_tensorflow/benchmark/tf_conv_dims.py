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
# Utility functions for computing convolution sizes.
#
# Implmented according to the doc:
# https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution
#
# For the 'SAME' padding, the output height and width are computed as:
# ```
# out_height = ceil(float(in_height) / float(strides[1]))
# out_width  = ceil(float(in_width) / float(strides[2]))
# ```
# and the padding on the top and left are computed as:
# ```
# pad_along_height = ((out_height - 1) * strides[1] + filter_height - in_height)
# pad_along_width = ((out_width - 1) * strides[2] + filter_width - in_width)
# pad_top = pad_along_height / 2
# pad_left = pad_along_width / 2
# ```
#
# For the 'VALID' padding, the output height and width are computed as:
# ```
# out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
# out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
# ```
from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf


def _check_strides(strides):
    """
    Validates strides parameters.

    :param strides:  [list]    List of 4 int or a Tensor of 4 elements. Convolution stride size.

    :returns:        [list]    List of 4 int or a Tensor of 4 elements, if inputs are valid.
    """
    if type(strides) == list or type(strides) == tuple:
        assert len(strides) == 4, 'Expect `strides` a list/tuple of length 4.'
        assert strides[0] == strides[3] == 1, 'Expect first and last dimension of `strides` = 1.'
    elif type(strides) == tf.Tensor:
        assert len(strides.get_shape()) == 1, 'Expect `strides` a rank 1 Tensor.'
        assert int(strides.get_shape()[0]) == 4, 'Expect `strides` to have 4 elements.'
        assert_strides = tf.assert_equal(
            tf.stack([strides[0], strides[3]]),
            tf.constant([1, 1], dtype=strides.dtype),
            message='Expect first and last dimension of `strides` = 1.')
        with tf.control_dependencies([assert_strides]):
            strides = tf.cast(strides, tf.int32)
    else:
        assert False, '`strides` has unknown type: {}'.format(type(strides))
    return strides


def _check_ksize(ksize):
    """
    Validates ksize parameters.

    :param ksize:    [list]    List of 4 int or a Tensor of 4 elements. Convolution kernel size.

    :returns:        [list]    List of 4 int or a Tensor of 4 elements, if inputs are valid.
    """
    if type(ksize) == list or type(ksize) == tuple:
        assert len(ksize) == 4, 'Expect `ksize` a list/tuple of length 4.'
    elif type(ksize) == tf.Tensor:
        assert len(ksize.get_shape()) == 1, 'Expect `ksize` a rank 1 Tensor.'
        assert int(ksize.get_shape()[0]) == 4, 'Expect `ksize` to have 4 elements.'
        ksize = tf.cast(ksize, tf.int32)
    else:
        assert False, '`ksize` has unknown type: {}'.format(type(ksize))
    return ksize


def calc_padding_4d(in_shape, ksize, strides, padding):
    """
    Calculates padding width on four dimensions: top, bottom, left, and right.

    :param x:        [Tensor]  Input tensor.
    :param ksize     [list]    List of 4 int or a Tensor of 4 elements. Convolution kernel size.
    :param strides   [list]    List of 4 int or a Tensor of 4 elements. Convolution stride size.
    :param padding   [list]    Padding method, `VALID` or `SAME`.

    :return          [tuple]   Tuple of 4 int. Padding length on top, bottom, left, and right.
    """
    ksize = _check_ksize(ksize)
    strides = _check_strides(strides)
    if padding == 'VALID':
        return 0, 0, 0, 0
    elif padding == 'SAME':
        if type(in_shape[1]) == int:
            out_size_h = calc_out_size_1d_np(in_shape[1], ksize[0], strides[1], padding)
            out_size_w = calc_out_size_1d_np(in_shape[2], ksize[1], strides[2], padding)
        elif type(in_shape[1]) == tf.Tensor:
            out_size_h = calc_out_size_1d(in_shape[1], ksize[0], strides[1], padding)
            out_size_w = calc_out_size_1d(in_shape[2], ksize[1], strides[2], padding)
        else:
            raise ValueError('Unknown type \"{}\"'.format(type(in_shape[1])))
        pad_h = calc_padding_1d(in_shape[1], out_size_h, ksize[0], strides[1], padding)
        pad_w = calc_padding_1d(in_shape[2], out_size_w, ksize[1], strides[2], padding)
        if type(pad_h) == int:
            pad_h0, pad_h1 = _div_padding_np(pad_h)
            pad_w0, pad_w1 = _div_padding_np(pad_w)
        elif type(pad_h) == tf.Tensor:
            pad_h0, pad_h1 = _div_padding(pad_h)
            pad_w0, pad_w1 = _div_padding(pad_w)
        else:
            raise ValueError('Unknown type \"{}\"'.format(type(pad_h)))
        return pad_h0, pad_h1, pad_w0, pad_w1
    else:
        raise ValueError('Unknown padding method \"{}\"'.format(padding))


def calc_padding_1d(in_size, out_size, ksize, stride, padding):
    """
    Calculates padding width on one dimension.

    :param in_size:  [Tensor]  Scalar. Input size.
    :param out_size: [Tensor]  Scalar. Output size.
    :param ksize:    [Tensor]  Scalar or int. Kernel size.
    :param strides:  [Tensor]  Scalar or int. Stride size.
    :param padding:  [string]  Padding method, `SAME` or `VALID`.

    :returns:        [Tensor]  Scalar. Padding size.
    """
    if padding == 'VALID':
        return 0
    elif padding == 'SAME':
        _pad = (out_size - 1) * stride + ksize - in_size
        if type(_pad) == int:
            return max(_pad, 0)
        elif type(_pad) == tf.Tensor:
            return tf.maximum(_pad, 0)
        else:
            raise ValueError('Unknown type \"{}\"'.format(type(_pad)))
    else:
        raise ValueError('Unknown padding method \"{}\"'.format(padding))


def _div_padding(pad_size):
    """
    Divides padding to two sides so that the features are centered.

    :param pad_size: [Tensor]  Scalar. Padding size.

    :return          [Tensor]  Scalar. First padding size.
    :return          [Tensor]  Scalar. Second padding size.
    """
    return tf.cast(tf.floor(tf.to_float(pad_size) / 2.0), tf.int32), tf.cast(
        tf.ceil(tf.to_float(pad_size) / 2.0), tf.int32)


def _div_padding_np(pad_size):
    """
    Divides padding to two sides so that the features are centered.

    :param pad_size: [np.ndarray]  Scalar. Padding size.

    :return          [int]  Scalar. First padding size.
    :return          [int]  Scalar. Second padding size.
    """
    return int(np.floor(float(pad_size) / 2.0)), int(np.ceil(float(pad_size) / 2.0))


def calc_out_size_4d(in_shape, ksize, strides, padding):
    """Calculates output shape (rank 4) of a 2D convolution operation.

    :param in_shape: [list]    Input tensor shape.
    :param ksize:    [list]    Kernel shape.
    :param strides:  [list]    Strides list.
    :param padding:  [string]  Padding method, `SAME` or `VALID`.

    :return          [list]    Output tensor shape.
    """
    strides = _check_strides(strides)
    ksize = _check_ksize(ksize)
    return tf.stack([
        in_shape[0],
        calc_out_size_1d(in_shape[1], ksize[0], strides[1], padding),
        calc_out_size_1d(in_shape[2], ksize[1], strides[2], padding), ksize[3]
    ])


def calc_out_size_1d(in_size, ksize, stride, padding):
    """
    Calculates output size on one dimension.

    :param in_size:  [int]     Input size.
    :param ksize:    [int]     Kernel size.
    :param stride:   [int]     Stride size.
    :param pad:      [string]  Padding method, `SAME` or `VALID`.

    :return          [int]     Output size.
    """

    if padding == 'VALID':
        return tf.cast(tf.ceil(tf.to_float(in_size - ksize + 1) / tf.to_float(stride)), tf.int32)
    elif padding == 'SAME':
        return tf.cast(tf.ceil(tf.to_float(in_size) / tf.to_float(stride)), tf.int32)
    else:
        raise ValueError('Unknown padding method \"{}\"'.format(padding))


def calc_out_size_1d_maxpool(in_size, ksize, stride, padding):
    """
    Calculates output size on one dimension.

    :param in_size:  [int]     Input size.
    :param ksize:    [int]     Kernel size.
    :param stride:   [int]     Stride size.
    :param pad:      [string]  Padding method, `SAME` or `VALID`.

    :return          [int]     Output size.
    """

    if padding == 'VALID':
        return tf.cast(tf.ceil(tf.to_float(in_size - ksize + 1) / tf.to_float(stride)), tf.int32)
    elif padding == 'SAME':
        return tf.cast(tf.ceil(tf.to_float(in_size) / tf.to_float(stride)), tf.int32)
    else:
        raise ValueError('Unknown padding method \"{}\"'.format(padding))


def calc_out_size_4d_np(in_shape, ksize, strides, padding):
    """Calculates output shape (rank 4) of a 2D convolution operation.

    :param in_shape: [list]    Input tensor shape.
    :param ksize:    [list]    Kernel shape.
    :param strides:  [list]    Strides list.
    :param padding:  [string]  Padding method, `SAME` or `VALID`.

    :return          [list]    Output tensor shape.
    """
    strides = _check_strides(strides)
    ksize = _check_ksize(ksize)
    return [
        in_shape[0],
        calc_out_size_1d_np(in_shape[1], ksize[0], strides[1], padding),
        calc_out_size_1d_np(in_shape[2], ksize[1], strides[2], padding), ksize[3]
    ]


def calc_out_size_1d_np(in_size, ksize, stride, padding):
    """
    Calculates output size on one dimension.

    :param in_size:  [int]     Input size.
    :param ksize:    [int]     Kernel size.
    :param stride:   [int]     Stride size.
    :param pad:      [string]  Padding method, `SAME` or `VALID`.

    :return          [int]     Output size.
    """

    if padding == 'VALID':
        return int(np.ceil(float(in_size - ksize + 1) / float(stride)))
    elif padding == 'SAME':
        return int(np.ceil(float(in_size) / float(stride)))
    else:
        raise ValueError('Unknown padding method \"{}\"'.format(padding))
