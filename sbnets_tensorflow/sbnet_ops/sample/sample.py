"""

   Sparse Blocks Networks
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
# A minimal sample implementing a single sparse convolution layer with synthetic data using SBNet primitives.
#

import numpy as np
import tensorflow as tf

sbnet_module = tf.load_op_library('../libsbnet.so')

def divup(a, b):
    return (a+b-1) // b

# Specify input tensor dimensions and block-sparsity parameters
batch = 4
hw = 256
channels = 64
blockSize = [16, 16]
blockStride = [14, 14]
blockOffset = [0, 0]
blockCount = [divup(hw, blockStride[0]), divup(hw, blockStride[1])]

# build kwargs to simplify op calls
inBlockParams = { "bsize": blockSize, "boffset": blockOffset, "bstride": blockStride }
outBlockParams = { "bsize": [blockSize[0]-2, blockSize[1]-2], "boffset": blockOffset, "bstride": blockStride }

# create a random mask representing attention/a priori sparsity
# threshold the mask to a specified percentile sparsity
mask = np.random.randn(batch, blockCount[0], blockCount[1], channels).astype(np.float32)
threshold = np.percentile(mask, 90)
sparseMask = np.greater(mask, threshold).astype(np.float32)

# upsample the mask to full resolution
upsampledMask = sparseMask.repeat(blockStride[0], axis=1).repeat(blockStride[1], axis=2)

# create a random input tensor
x = tf.constant( np.random.randn(batch, hw, hw, channels).astype(np.float32) )

# create a random weight tensor
w = tf.constant( np.random.randn(3, 3, channels, channels).astype(np.float32) )

# reduce the mask to indices by using a fused pooling+indexing operation
indices = sbnet_module.reduce_mask(mask, blockCount, tol=0.5, **inBlockParams)

# stack active overlapping tiles to batch dimension
blockStack = sbnet_module.sparse_gather(
    x, indices.bin_counts, indices.active_block_indices, transpose=True, **inBlockParams)

# perform dense convolution on a sparse stack of tiles
convBlocks = tf.nn.conv2d(
    blockStack, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')

# write/scatter the tiles back on top of original tensor
# note that the output tensor is reduced by 1 on each side due to 'VALID' convolution
validX = x[:, 1:hw-1, 1:hw-1, :]
y = sbnet_module.sparse_scatter(
    convBlocks, indices.bin_counts, indices.active_block_indices,
    validX, transpose=True, add=False, atomic=False, **outBlockParams)

sess = tf.Session()
y_output, = sess.run([y])


