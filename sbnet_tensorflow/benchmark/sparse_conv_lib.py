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
# Sparse convolution operators.
#
# Usage:
# ```
# import numpy as np
# import tensorflow as tf
#
# from sparse_conv_lib import convert_mask_to_block_indices, sparse_conv2d
#
# # Binary mask to define sparsity.
# mask = tf.constant(
#     np.array(
#         [[
#             [0, 0, 0, 0, 0],    # YAPF_NO_FORMAT
#             [0, 0, 1, 0, 0],
#             [1, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0]
#         ]],
#         dtype=np.float32))
# # Convert binary mask to block representation.
# ind_blk = convert_mask_to_block_indices(mask, [1, 3, 3, 1], [1, 1, 1, 1], [3, 3, 1, 1],
#                                         [1, 1, 1, 1], 'SAME', .1)
#
# # Sparse convolution.
# x = tf.constant(np.ones([1, 5, 5, 1], dtype=np.float32))
# w = tf.constant(np.ones([3, 3, 1, 1], dtype=np.float32))
# y = sparse_conv2d(x, w, ind_blk, [1, 1, 1, 1], 'SAME')
#
# with tf.Session():
#     print(np.squeeze(y.eval()))
#
# >> Output
# >> [[ 0.  6.  6.  6.  0.]
#     [ 6.  9.  9.  9.  0.]
#     [ 6.  9.  9.  9.  0.]
#     [ 6.  9.  0.  0.  0.]
#     [ 0.  0.  0.  0.  0.]]
# ```
from __future__ import division, print_function

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from collections import namedtuple

import logger
from tf_conv_dims import calc_padding_4d, calc_out_size_4d, calc_out_size_4d_np

log = logger.get()

sbnet_module = tf.load_op_library('../sbnet_ops/libsbnet.so')

BlockParams = namedtuple('BlockParams', ['bsize', 'bsize_out', 'boffset', 'bcount', 'bstrides'])

# Gradients registration.
@ops.RegisterGradient("SparseGather")
def _sparse_gather_grad(op, grad):
    # x is shaped like full tensor [NHWC]
    # grad is shaped as gathered blocks [Nblocks*BH*BW*C]
    x = op.inputs[0]
    binCounts = op.inputs[1]
    activeBlockIndices = op.inputs[2]
    bsize = op.get_attr("bsize")
    bstride = op.get_attr("bstride")
    boffset = op.get_attr("boffset")
    transpose = op.get_attr("transpose")

    # if scatter is overlapping then gradient still work because we use atomic adds
    # compute dOutput/dx
    result = sbnet_module.sparse_scatter(
        grad,
        binCounts,
        activeBlockIndices,
        tf.zeros_like(x),
        bsize=bsize,
        bstride=bstride,
        boffset=boffset,
        add=True,
        transpose=transpose,
        atomic=True)

    return [result, None, None]    # no gradient wrt indices


@ops.RegisterGradient("SparseScatter")
def _sparse_scatter_grad(op, grad):
    # x is shaped like blocked tensor of gathered blocks [Nblocks*BH*BW*C]
    # grad is shaped as output tensor [NHWC]
    blocksX = op.inputs[0]
    binCounts = op.inputs[1]
    activeBlockIndices = op.inputs[2]
    ybase = op.inputs[3]
    bsize = op.get_attr("bsize")
    bstride = op.get_attr("bstride")
    boffset = op.get_attr("boffset")
    doAdd = op.get_attr("add")
    transpose = op.get_attr("transpose")

    dout_dx = sbnet_module.sparse_gather(
        grad,
        binCounts,
        activeBlockIndices,
        bsize=bsize,
        bstride=bstride,
        boffset=boffset,
        transpose=transpose)

    # return a list of gradients of output with respect to each input
    if not doAdd:
        # scatter blocks of zeroes over a base tensor of ones to compute a stamp-out gradient mask for dy_dybase
        stamp_out_blocks = sbnet_module.sparse_scatter(
            tf.zeros_like(blocksX),
            binCounts,
            activeBlockIndices,
            tf.ones_like(grad),
            bsize=bsize,
            bstride=bstride,
            boffset=boffset,
            add=False,
            transpose=transpose)
        dy_dybase = grad * stamp_out_blocks
        return [dout_dx, None, None, dy_dybase]
    else:
        # d(x+ybase)/dybase = 1, so just pass back grad as dout_dybase
        return [dout_dx, None, None, grad]


def _pad_input(x, ksize, strides, padding, bsize=None, bstrides=None):
    """Pads the input tensor.
    Optional to pass in block strides. The right hand side padding will be increased if the last
    block does not fit in (no effect on the convolution results.

    :param x:        [Tensor]   [N, H, W, C]. input tensor, dtype float32.
    :param ksize:    [list]     List of 4 int. Sparse convolution kernel size.
    :param strides:  [list]     List of 4 int. Sparse convolution stride size.
    :param padding:  [string]   `VALID` or `SAME`, padding method for sparse convolution.
    :param bsize     [list]     List of 4 int. Block size. Optional.
    :param bstrides: [list]     List of 4 int. Block strides. Optional.

    :return          [Tensor]   [N, H+Ph, W+Pw, C]. Padded input tensor.
    """
    x_shape = tf.shape(x)
    if padding == 'SAME':
        pad_h0, pad_h1, pad_w0, pad_w1 = calc_padding_4d(x_shape, ksize, strides, padding)

        if bstrides is not None:
            # Here we do not use the standard padding on the right hand side.
            # If the convolution results is larger than expected, the scatter function will not use
            # out-of-boundary points.
            assert bsize is not None, 'Must pass in bsize and bstrides together.'
            h = x_shape[1] + pad_h0 + pad_h1
            w = x_shape[2] + pad_w0 + pad_w1
            pad_h1 += tf.mod(-h + bsize[1], bstrides[1])
            pad_w1 += tf.mod(-w + bsize[2], bstrides[2])
        return tf.pad(x, [[0, 0], [pad_h0, pad_h1], [pad_w0, pad_w1], [0, 0]])
    else:
        if bstrides is not None:
            assert bsize is not None, 'Must pass in bsize and bstrides together.'
            h = x_shape[1]
            w = x_shape[2]
            pad_h1 = tf.mod(-h + bsize[1], bstrides[1])
            pad_w1 = tf.mod(-w + bsize[2], bstrides[2])
            return tf.cond(
                tf.logical_or(tf.greater(pad_h1, 0), tf.greater(pad_w1, 0)),
                lambda: tf.pad(x, [[0, 0], [0, pad_h1], [0, pad_w1], [0, 0]]), lambda: x)
        else:
            return x


def _get_offset_array_tf(shape):
    """
    Computes the offset array used to upsample indices with TensorFlow.

    :param shape:   [list]     Window shape.
    """
    center = [(ss - 1) // 2 for ss in shape]
    axes = [tf.range(-cc, ss - cc, dtype=tf.int32) for cc, ss in zip(center, shape)]
    # Broadcast and match dimension.
    if len(shape) > 1:
        for jj in range(len(shape)):
            for ii in range(len(shape) + 1):
                if ii != jj:
                    axes[jj] = tf.expand_dims(axes[jj], ii)
        for jj in range(len(shape)):
            shape_ = [ss for ss in shape] + [1]
            shape_[jj] = 1
            axes[jj] = tf.tile(axes[jj], shape_)
        offset = tf.concat(axes, len(shape))
    return offset


def _get_offset_array(shape):
    """
    Computes the offset array used to upsample indices with NumPy (static).

    :param shape:   [list]     Window shape.
    """
    center = [int(ss - 1) // 2 for ss in shape]
    axes = [np.arange(-cc, int(ss) - cc).astype(np.int32) for cc, ss in zip(center, shape)]
    if len(shape) > 1:
        for jj in range(len(shape)):
            for ii in range(len(shape) + 1):
                if ii != jj:
                    axes[jj] = np.expand_dims(axes[jj], ii)
        for jj in range(len(shape)):
            shape_ = [int(ss) for ss in shape] + [1]
            shape_[jj] = 1
            axes[jj] = np.tile(axes[jj], shape_)
        offset = np.concatenate(axes, len(shape))
        return tf.constant(offset)
    else:
        return tf.constant(axes[0])


def _calc_block_strides(bsize, ksize, strides):
    """Calculates strides for blocks.

    :param bsize:     [list]        List of 4 int. Size of blocks, or downsample ratio.
    :param ksize:     [list]        List of 4 int. Sparse convolution kernel size.
    :param strides:   [list]        List of 4 int. Sparse convolution strides.

    :return           [list]        List of 4 int. Block strides.
    """
    return [1, bsize[1] - ksize[0] + strides[1], bsize[2] - ksize[1] + strides[2], 1]


def upsample_indices(indices, ksize, strides):
    """
    Upsamples the indices to have all indices in a rectangle.

    :param indices:   [Tensor]      [M, 3]. Center locations (N, H, W) of the M rectangles.
                                    Dtype int32.
    :param ksize:     [list]        Size of the rectangle, or downsample ratio.
    :param strides:   [list]        Strides of the pooling operation.

    :return           [Tensor]      [M, h, w, 3]. Locations of all pixels in the rectangles.
                                    Dtype int32.
    """
    assert len(indices.get_shape()) == 2, 'Expect indices rank = 2'
    assert ksize[0] == ksize[3] == 1, 'Expect first and last dimensions of ksize = 1'
    assert strides[0] == strides[3] == 1, 'Expect first and last dimensions of strides = 1, {}'.format(
        strides)
    h_scale = strides[1]
    w_scale = strides[2]
    scale = tf.stack([1, h_scale, w_scale])
    indices *= scale
    # Since we always use VALID to perform pooling, shift is needed here.
    shift = tf.stack([0, (ksize[1] - 1) // 2, (ksize[2] - 1) // 2])
    indices += shift
    indices_ = tf.expand_dims(tf.expand_dims(indices, 1), 2)
    # indices_ = tf.tile(indices_, [1, ksize[1], ksize[2], 1])
    offset = _get_offset_array(ksize[0:3])
    indices_ += offset
    return indices_


def convert_mask_to_indices(mask, bsize, ksize, strides, padding, tol):
    """
    Converts a binary mask to sparse indices.

    :param mask:     [Tensor]   [N, H, W]. 1 indicates non-sparse locations. Dtype float32.
    :param bsize:    [list]     List of 4 int. Size of blocks, or downsample ratio.
    :param ksize:    [list]     List of 4 int. Sparse convolution kernel size.
    :param strides:  [list]     List of 4 int. Sparse convolution stride size.
                                Currently only supports when,
                                1) (bsize[1] - ksize[0]) % strides[1] == 0 and,
                                2) (bsize[2] - ksize[1]) % strides[2] == 0
    :param padding:  [string]   `VALID` or `SAME`, padding method for sparse convolution.
    :param tol:      [float]    Lower bound of occupancy for creating a rectangle.

    :return          [Tensor]   [M, 3]. Center locations (N, H, W) of M rectangles. Dtype int32.
    """
    ERR_MSG_RANK = 'Expect mask rank = 3'
    ERR_MSG_DIV = 'Expect `stride` divides `bsize` - `ksize`. stride {}, bsize {}, ksize {}.'
    ERR_MSG_DIM = 'Expect first and last dimensions of strides = 1. Dim {}.'

    assert len(mask.get_shape()) == 3, ERR_MSG_RANK
    assert type(bsize) in [list, tuple], '`bsize` needs to be a list or tuple.'
    assert type(ksize) in [list, tuple], '`ksize` needs to be a list or tuple.'
    assert type(strides) in [list, tuple], '`strides` needs to be a list or tuple.'
    assert (bsize[1] - ksize[0]) % strides[1] == 0, ERR_MSG_DIV.format(
        strides[1], bsize[1], ksize[0])
    assert (bsize[2] - ksize[1]) % strides[2] == 0, ERR_MSG_DIV.format(
        strides[2], bsize[2], ksize[1])
    assert strides[0] == strides[3] == 1, ERR_MSG_DIM.format(strides)

    bstrides = _calc_block_strides(bsize, ksize, strides)

    # Pad mask.
    mask_ = tf.expand_dims(mask, 3)
    mask_ = _pad_input(mask_, ksize, strides, padding, bsize=bsize, bstrides=bstrides)
    mask_ = tf.nn.max_pool(mask_, bsize, bstrides, 'VALID')    # Blocks are always valid conv.
    mask_ = tf.squeeze(mask_, [3])
    indices = tf.where(tf.greater(mask_, tol))
    indices = tf.cast(indices, tf.int32)
    return indices


def convert_mask_to_block_indices(mask, bsize, ksize, strides, padding, tol):
    """
    Converts a binary mask to block sparse indices.

    :param mask:     [Tensor]   [N, H, W]. 1 indicates non-sparse locations. Dtype float32.
    :param bsize:    [list]     List of 4 int. Size of blocks, or downsample ratio.
    :param ksize:    [list]     List of 4 int. Sparse convolution kernel size.
    :param strides:  [list]     List of 4 int. Sparse convolution stride size.
                                Currently only supports when,
                                1) (bsize[1] - ksize[0]) % strides[1] == 0 and,
                                2) (bsize[2] - ksize[1]) % strides[2] == 0
    :param padding:  [string]   `VALID` or `SAME`, padding method for sparse convolution.
    :param tol:      [float]    Lower bound of occupancy for creating a rectangle.

    :return          [Tensor]   [M, h, w, 3]. Pixel locations of M rectangles. Dtype int32.
    """
    indices = convert_mask_to_indices(mask, bsize, ksize, strides, padding, tol)
    bstrides = _calc_block_strides(bsize, ksize, strides)
    blk_indices = upsample_indices(indices, bsize, bstrides)
    return blk_indices


def calc_block_params(in_size, bsize, ksize, strides, padding, static=True):
    """
    Calculates block parameters for a single convolution layer.

    :param in_size:  [list]     List of 4 int. Size of the convolution input.
    :param bsize:    [list]     List of 4 int. Size of blocks, or downsample ratio.
    :param ksize:    [list]     List of 4 int. Sparse convolution kernel size.
    :param strides:  [list]     List of 4 int. Sparse convolution stride size.
                                Currently only supports when,
                                1) (bsize[1] - ksize[0]) % strides[1] == 0 and,
                                2) (bsize[2] - ksize[1]) % strides[2] == 0
    :param padding:  [string]   `VALID` or `SAME`, padding method for sparse convolution.

    :return          [tuple]
        bsize:
        bsize_out:
        boffset:
        bcount:
        bstrides:
    """

    assert ((bsize[1] - ksize[0]) % strides[1] == 0)
    assert ((bsize[2] - ksize[1]) % strides[2] == 0)

    bstrides = _calc_block_strides(bsize, ksize, strides)
    pad_h0, pad_h1, pad_w0, pad_w1 = calc_padding_4d(in_size, ksize, strides, padding)
    h = in_size[1]
    w = in_size[2]
    # Make padding divides blocks.
    pad_h1 += (-h + bsize[1]) % bstrides[1]
    pad_w1 += (-w + bsize[2]) % bstrides[2]
    boffset = [-pad_h0, -pad_w0]
    x_pad_shape = [
        in_size[0], in_size[1] + pad_h0 + pad_h1, in_size[2] + pad_w0 + pad_w1, in_size[3]
    ]
    if static:
        out_shape = calc_out_size_4d_np(x_pad_shape, [bsize[1], bsize[2], 1, 1], bstrides, 'VALID')
    else:
        out_shape = calc_out_size_4d(x_pad_shape, [bsize[1], bsize[2], 1, 1], bstrides, 'VALID')
    bcount = [out_shape[1], out_shape[2]]
    bsize_out = calc_out_size_4d_np(bsize, ksize, strides, 'VALID')
    bsize = bsize[1:3]
    bstrides = bstrides[1:3]
    bsize_out = bsize_out[1:3]
    # print('h w', h, w)
    # print('bcount', bcount)
    # print('bsize', bsize)
    # print('bsize_out', bsize_out)
    # print('boffset', boffset)
    # print('bstrides', bstrides)
    # print(pad_h0, pad_w0, boffset)
    if static:
        assert (pad_h0 == -boffset[0])
        assert (pad_w0 == -boffset[1])
    for i, siz in zip([0, 1], [h, w]):
        # make sure last block is inside
        err_msg = 'Making sure last block is inside boffset {} bstrides {} bcount {} size {}'.format(
            boffset[i], bstrides[i], bcount[i], siz)
        assert (boffset[i] + bstrides[i] * (bcount[i] - 1) < siz), err_msg
    return BlockParams(
        bsize=bsize, bsize_out=bsize_out, boffset=boffset, bcount=bcount, bstrides=bstrides)


def calc_block_params_res_block(in_size, bsize, ksize_list, strides, padding):
    """
    Calculates block parameters for a residual block.

    :param in_size:  [list]     List of 4 int. Size of the residual block input.
    :param bsize:    [list]     List of 4 int. Size of blocks, or downsample ratio, for each
                                convolution layer in the residual block.
    :param ksize:    [list]     List of list of 4 int. Sparse convolution kernel size.
    :param strides:  [list]     List of 4 int. Sparse convolution stride size, for the first
                                convolution in the residual block.
                                Currently only supports when,
                                1) (bsize[1] - ksize[0]) % strides[1] == 0 and,
                                2) (bsize[2] - ksize[1]) % strides[2] == 0
    :param padding:  [string]   `VALID` or `SAME`, padding method for sparse convolution.

    :return
    """
    # Use the receptive field as the kernel size.
    ksize_h = 1 + sum([kk[0] - 1 for kk in ksize_list])
    ksize_w = 1 + sum([kk[1] - 1 for kk in ksize_list])
    ksize_real = [ksize_h, ksize_w, 1, 1]
    return calc_block_params(in_size, bsize, ksize_real, strides, padding)


def convert_mask_to_indices_custom(mask, block_params, tol, avgpool=False):
    """
    Converts a binary mask to sparse index format for custom CUDA kernel and TF ops.

    :param mask:         [Tensor]   [N, H, W]. 1 indicates non-sparse locations. Dtype float32.
    :param block_params  [tuple]    Contains bsize, boffset, bcount, bstrides.
    :param tol:          [float]    Lower bound of occupancy for creating a rectangle.

    :return          [tuple]
        bin_counts:           [Tensor]. Number of active locations for each bin.
        active_block_indices: [Tensor]. [M]. Center locations of M rectangles. Dtype int64.
    """
    return sbnet_module.reduce_mask(
        mask, block_params.bcount,
        bsize=block_params.bsize,
        boffset=block_params.boffset,
        bstride=block_params.bstrides,
        avgpool=avgpool,
        tol=tol)


def sparse_conv2d(x, w, blk_indices, strides, padding):
    """
    Performs 2D convolution on a sparse feature map, given indices.
    Naive python implementation of sparse convolution using gather and scatter.

    :param x:           [Tensor]  [N, H, W, C]. Input activation tensor, dtype float32.
    :param w:           [Tensor]  [I, J, C, K]. Convolution kernel, dtype float32.
    :param blk_indices: [Tensor]  [M, h, w, 3]. Block indices of rectangles.
    :param strides:     [list]    List of 4 int, convolution strides.
    :param padding:     [string]  `VALID` or `SAME`, padding method for sparse convolution.

    :return             [Tensor]  [N, H', W', C]. Convolution results.
    """
    blk_shape = tf.shape(blk_indices)
    blk_indices_ = tf.reshape(blk_indices, [-1, 3])
    ksize = tf.shape(w)

    # Calculate the block strides.
    bstrides = _calc_block_strides(blk_shape, ksize, strides)

    # Calculate the output size.
    x_shape = tf.shape(x)
    out_shape = calc_out_size_4d(x_shape, ksize, strides, padding)

    # Pad input.
    x_ = _pad_input(
        x, ksize, strides, padding, bsize=[1, blk_shape[1], blk_shape[2], 1], bstrides=bstrides)

    # Convolution when number of indices is larger than zero.
    def _conv_nonzero():
        # Gather patches.
        p = tf.gather_nd(x_, blk_indices_)

        # Reshape patches.
        p = tf.reshape(p, [blk_shape[0], blk_shape[1], blk_shape[2], -1])

        # Convolution on patches.
        q = tf.nn.conv2d(p, w, strides, 'VALID', use_cudnn_on_gpu=True)

        # Paste convolution results.
        q_shape = tf.shape(q)

        def _strides_gt_one():
            # Calculate output indices when strides > 1.
            blk_indices_crop = tf.strided_slice(blk_indices, [0, 0, 0, 0], [
                blk_shape[0], q_shape[1] * strides[1], q_shape[2] * strides[2], 3
            ], strides)
            blk_indices_crop = blk_indices_crop // tf.stack([1, strides[1], strides[2]])
            return blk_indices_crop

        def _strides_one():
            # Calculate otuput indices when strides = 1.
            return blk_indices[:, :q_shape[1], :q_shape[2], :]

        strides_gt_one = tf.logical_or(tf.greater(strides[1], 1), tf.greater(strides[2], 1))
        blk_indices_crop = tf.cond(strides_gt_one, _strides_gt_one, _strides_one)
        y = tf.scatter_nd(blk_indices_crop, q, out_shape)
        return y

    return tf.cond(
        tf.equal(tf.size(blk_indices_), 0), lambda: tf.zeros(out_shape, dtype=x.dtype),
        _conv_nonzero)


def cuda_timer_start_op(name):
    return sbnet_module.cuda_op_timer(timer_name=name, is_start=True)


def cuda_timer_end_op(name):
    return sbnet_module.cuda_op_timer(timer_name=name, is_start=False)


def sparse_conv2d_custom(x,
                         w,
                         indices,
                         block_params,
                         strides,
                         use_var=False,
                         transpose=False,
                         atomic=False):
    assert strides[1] == strides[2] == 1, 'Only accept strides=1'
    # TODO: make the gather op also accepting a Tensor for bsize, ksize, etc.
    ksize = [int(ss) for ss in w.get_shape()]
    p = sbnet_module.sparse_gather(
        x,
        indices.bin_counts,
        indices.active_block_indices,
        bsize=block_params.bsize,
        boffset=block_params.boffset,
        bstride=block_params.bstrides,
        transpose=transpose)

    # Convolution on patches.
    if transpose:
        q = tf.nn.conv2d(p, w, strides, 'VALID', data_format='NCHW', use_cudnn_on_gpu=True)
    else:
        q = tf.nn.conv2d(p, w, strides, 'VALID', use_cudnn_on_gpu=True)

    # Allocate output tensor.
    if use_var:
        y = sbnet_module.sparse_scatter_var(
            q,
            indices.bin_counts,
            indices.active_block_indices,
            x,
            bsize=block_params.bsize_out,
            boffset=[0, 0],
            bstride=block_params.bstrides,
            add=False,
            transpose=transpose,
            atomic=atomic)
    else:
        y = sbnet_module.sparse_scatter(
            q,
            indices.bin_counts,
            indices.active_block_indices,
            x,
            bsize=block_params.bsize_out,
            boffset=[0, 0],
            bstride=block_params.bsize_out,
            add=False,
            transpose=transpose,
            atomic=atomic)
    return y


def _batch_norm(name, x, is_training, data_format='NHWC'):
    """
    Applies batch normalization.

    :param name:       [string]    Name of the variable scope.
    :param x:          [Tensor]    Tensor to apply BN on.
    :param is_training [bool]   Whether in training mode.

    :return:           [Tensor]    Normalized activation.
    """
    bn = tf.contrib.layers.batch_norm(
        x, fused=True, scale=True, data_format=data_format, is_training=is_training, scope=name)
    return bn
    # log.warning('Not using BN to test performance at inference time')
    # return x


def _relu(name, x):
    """
    Applies ReLU function.

    :param name: [string]     Name of the op.
    :param x:    [Tensor]     Input to the function.

    :return:     [Tensor]     Output of the function.
    """
    return tf.nn.relu(x, name=name)
    # log.warning('Not using ReLU to test performance at inference time')
    # return x


def _stride_arr(n, data_format='NHWC'):
    """Makes strides array for downsampling convolution."""
    if data_format == 'NHWC':
        return [1, n, n, 1]
    elif data_format == 'NCHW':
        return [1, 1, n, n]
    else:
        raise ValueError('Unknown data format: {}'.format(data_format))


def _conv(name,
          x,
          ksize,
          strides,
          padding,
          data_format='NHWC',
          weight_decay=None,
          dtype=tf.float32,
          weights_on_cpu=False):
    """
    Convolution layer.

    :param name           [string]     Name of the op.
    :param x:             [Tensor]     Input to the downsample.
    :param ksize          [list]       4-D kernel shape.
    :param strides:       [list]       4-D strides array.
    :param padding:       [string]     Convolution padding strategy.
    :param data_format:   [string]     'NHWC' or 'NCHW'.

    :return:              [Tensor]     Convolution output.
    """
    with tf.variable_scope(name):
        in_filters = ksize[2]
        out_filters = ksize[3]
        n = ksize[0] * ksize[1] * out_filters
        init = tf.truncated_normal_initializer(
            mean=0.0, stddev=np.sqrt(2.0 / n), seed=0, dtype=dtype)

        def _reg(x):
            if weight_decay is not None:
                return tf.multiply(tf.nn.l2_loss(x), weight_decay)
            else:
                return None

        if weight_decay is not None:
            reg = _reg
        else:
            reg = None

        kernel = tf.get_variable(
            'w', ksize, initializer=init, regularizer=reg, dtype=dtype, trainable=True)

        return tf.nn.conv2d(
            x, kernel, strides, padding, data_format=data_format, use_cudnn_on_gpu=True)


def _bottleneck_residual(x,
                         ksize_list,
                         strides,
                         padding,
                         is_training,
                         data_format='NHWC',
                         no_activation=False):
    with tf.variable_scope('sub1'):
        if not no_activation:
            x = _batch_norm('bn1', x, is_training, data_format)
            x = _relu('relu1', x)

        STRIDES_ERR_MSG = 'Strides height and width are not the same.'
        if data_format == 'NHWC':
            assert strides[1] == strides[2], STRIDES_ERR_MSG
        elif data_format == 'NCHW':
            assert strides[2] == strides[3], STRIDES_ERR_MSG
        x = _conv(
            'conv1',
            x,
            ksize_list[0],
            _stride_arr(strides[2], data_format),
            padding,
            data_format=data_format)

    with tf.variable_scope('sub2'):
        x = _batch_norm('bn2', x, is_training, data_format)
        x = _relu('relu2', x)
        x = _conv(
            'conv2',
            x,
            ksize_list[1],
            _stride_arr(1, data_format),
            padding,
            data_format=data_format)

    with tf.variable_scope('sub3'):
        x = _batch_norm('bn3', x, is_training, data_format)
        x = _relu('relu3', x)
        x = _conv(
            'conv3',
            x,
            ksize_list[2],
            _stride_arr(1, data_format),
            padding,
            data_format=data_format)
    return x


def res_block_bottleneck(x,
                         ksize_list,
                         strides,
                         is_training,
                         data_format='NHWC',
                         w_project=None,
                         no_activation=False):
    """
    Computes y = x + F(x), where F(x) is the residual block function. At downsample layers, applies
    a downsample function on x as well.
    """
    if w_project is not None:
        x_ = tf.conv2d(x, w_project, strides, padding='SAME', data_format=data_format)
    else:
        x_ = x
    return x_ + _bottleneck_residual(
        x,
        ksize_list,
        strides,
        'SAME',
        is_training,
        data_format=data_format,
        no_activation=no_activation)


def sparse_res_block_bottleneck(x,
                                ksize_list,
                                indices,
                                block_params,
                                strides,
                                is_training,
                                data_format='NHWC',
                                w_project=None,
                                no_activation=False,
                                use_var=False):
    """
    Computes y = x + F(x), where F(x) is the residual block function. At downsample layers, applies
    a downsample function on x as well.

    :param x:                 [Tensor]  [N, H, W, C]. Input activation tensor, dtype float32.
    :param ksize_list:        [list]    List of list of 4 int. Kernel size for each convolution
                                        layer in the residual block.
    :param indices:           [tuple]   Non-sparse locations returned by reduce_mask.
    :param block_params:      [tuple]   BlockParam namedtuple.
    :param

    :return
    """
    transpose = True if data_format == 'NCHW' else False
    p = sbnet_module.sparse_gather(
        x,
        indices.bin_counts,
        indices.active_block_indices,
        bsize=block_params.bsize,
        boffset=block_params.boffset,
        bstride=block_params.bstrides,
        transpose=transpose)

    if w_project is not None:
        x = tf.conv2d(x, w_project, strides, padding='SAME')

    # Set shape for BN in the residual function.
    if transpose:
        p.set_shape([None, x.get_shape()[3], block_params.bsize[0], block_params.bsize[1]])
    else:
        p.set_shape([None, block_params.bsize[0], block_params.bsize[1], x.get_shape()[3]])

    q = _bottleneck_residual(
        p,
        ksize_list,
        strides,
        'VALID',
        is_training,
        data_format=data_format,
        no_activation=no_activation)

    if use_var:
        y = sbnet_module.sparse_scatter_var(
            q,
            indices.bin_counts,
            indices.active_block_indices,
            x,
            bsize=block_params.bsize_out,
            boffset=[0, 0],
            bstride=block_params.bsize_out,
            add=True,
            transpose=transpose)
    else:
        y = sbnet_module.sparse_scatter(
            q,
            indices.bin_counts,
            indices.active_block_indices,
            x,
            bsize=block_params.bsize_out,
            boffset=[0, 0],
            bstride=block_params.bsize_out,
            add=True,
            transpose=transpose)
    return y


def sparse_conv2d_matmul(x, w, blk_indices, strides, padding):
    """
    Performs 2D convolution using matrix multiplication on a sparse feature map.
    Naive python implementation of sparse convolution using gather and scatter.

    :param x:           [Tensor]  [N, H, W, C]. Input activation tensor, dtype float32.
    :param w:           [Tensor]  [I, J, C, K]. Convolution kernel, dtype float32.
    :param blk_indices: [Tensor]  [M, h, w, 3]. Block indices of rectangles.
    :param strides:     [list]    List of 4 int, convolution strides.
    :param padding:     [string]  `VALID` or `SAME`, padding method for sparse convolution.

    :return             [Tensor]  [N, H', W', C]. Convolution results.
    """
    blk_indices_ = tf.reshape(blk_indices, [-1, 3])
    blk_shape = tf.shape(blk_indices)
    ksize = tf.shape(w)

    # Calculate the block strides.
    bstrides = _calc_block_strides(blk_shape, ksize, strides)

    # Calculate the output size.
    x_shape = tf.shape(x)
    out_shape = calc_out_size_4d(x_shape, ksize, strides, padding)

    # Pad input.
    x_ = _pad_input(
        x, ksize, strides, padding, bsize=[1, blk_shape[1], blk_shape[2], 1], bstrides=bstrides)

    # In matrix multiplication mode, the block patch should be the same as the kernel size.
    assert_shape = tf.assert_equal(
        tf.stack([blk_shape[1], blk_shape[2]]),
        tf.stack([ksize[0], ksize[1]]),
        message='Expect blk_indices.shape[1] == w.shape[0] and blk_indices.shape[2] == w.shape[1].')

    # Currently we do not support strides > 1 in this matrix multiplication mode. Could be supported
    # in the future.
    assert_strides = tf.assert_equal(
        tf.cast(tf.stack([strides[1], strides[2]]), tf.int64),
        tf.constant([1, 1], dtype=tf.int64),
        message='Strides > 1 not supported.')

    # Convolution when number of indices is larger than zero.
    def _conv_nonzero():
        # Gather patches.
        p = tf.gather_nd(x_, blk_indices_)
        p_ = tf.reshape(p, [-1, ksize[0] * ksize[1] * ksize[2]])

        # Convolution on patches.
        w_ = tf.reshape(w, [ksize[0] * ksize[1] * ksize[2], -1])
        q = tf.matmul(p_, w_)

        # Center locations.
        blk_indices_crop = blk_indices[:, 0, 0, :]

        #  Project back to an image.
        y = tf.scatter_nd(blk_indices_crop, q, out_shape)
        return y

    with tf.control_dependencies([assert_shape, assert_strides]):
        return tf.cond(
            tf.equal(tf.size(blk_indices_), 0), lambda: tf.zeros(out_shape, dtype=x.dtype),
            _conv_nonzero)


def mask_conv2d(x, w, mask, strides, padding):
    """Masked 2D convolution. Used to check 2D sparse convolution.

    :param x:         [Tensor]    Convolution feature map, 4D, dtype float32.
    :param w:         [Tensor]    Convolution kernel, 4D, dtype float32.
    :param mask:      [Tensor]    Binary mask, 3D or 4D, [N, H, W] or [N, H, W, 1], dtype float32.
    :param strides:   [list]      List of 4 int. Convolution strides.
    :param padding:   [string]    Convolution padding method, `VALID` or `SAME`.
    """
    assert len(mask.get_shape()) in [3, 4], 'Mask shape must be 3D or 4D.'
    if len(mask.get_shape()) == 3:
        mask_ = tf.expand_dims(mask, 3)
    elif len(mask.get_shape()) == 4:
        mask_ = mask
        assert mask.get_shape()[-1] == 1, '4D mask last dimension must be 1.'
    ksize = [int(ss) for ss in w.get_shape()]
    psize = [1, ksize[0], ksize[1], 1]
    mask_ = tf.nn.max_pool(mask_, psize, strides, padding)
    return tf.nn.conv2d(x, w, strides, padding) * mask_
