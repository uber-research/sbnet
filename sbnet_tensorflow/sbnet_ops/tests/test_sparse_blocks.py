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


import os
import numpy as np
import unittest

from math import floor, ceil
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sbnet_module = tf.load_op_library('../libsbnet.so')


# Python implementation of gradients
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

    # if scatter is overlapping then gradient should still work
    # because we are overwriting the same values
    # compute dOutput/dx
    result = sbnet_module.sparse_scatter(
        grad,
        binCounts,
        activeBlockIndices,
        tf.zeros_like(x),    # output base tensor to add on top of
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

    dout_dx = sbnet_module.sparse_gather(
        grad, binCounts, activeBlockIndices, bsize=bsize, bstride=bstride, boffset=boffset)

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
            add=False)
        dy_dybase = grad * stamp_out_blocks
        return [dout_dx, None, None, dy_dybase]
    else:
        # d(x+ybase)/dybase = 1, so just pass back grad as dout_dybase
        return [dout_dx, None, None, grad]


def calcBlockCount1d(WW, SS, VV, BOFFSW):
    assert (WW >= SS)
    assert (SS >= 1)
    assert (VV >= 1)
    assert (BOFFSW < WW)
    #k = 0
    #while BOFFSW+k*VV+SS <= WW:
    #while k*VV <= WW-BOFFSW-SS:
    #    k += 1
    #return k
    pixelsOfLastBlock = 1  # set to SS to fully fit the last block
    return 1 + (WW - BOFFSW - pixelsOfLastBlock) // VV


def calcBlockCounts(HH, WW, RR, SS, UU, VV, BOFFSH, BOFFSW):
    # HH, WW = image sizes
    # RR, SS = block sizes
    # UU, VV = block strides
    # BOFFSH,W = block offsets (where block 0,0 starts)
    # Computes block counts to cover HH fully with padding if necessary, given offset, block size and block stride
    return calcBlockCount1d(HH, RR, UU, BOFFSH), calcBlockCount1d(WW, SS, VV, BOFFSW)


def generateInputs(inputSize, RS, UV, BOFFS, N):
    np.random.seed(0)
    NN, HH, WW, CC = N, inputSize[0], inputSize[1], inputSize[2]
    RR, SS = RS[0], RS[1]    # block sizes
    UU, VV = UV[0], UV[1]    # block strides
    BCH, BCW = calcBlockCounts(HH, WW, RR, SS, UU, VV, BOFFS[0], BOFFS[1])
    mask = np.random.randn(NN, HH, WW, 1).astype(np.float32)    # mask is NHW1
    x = np.random.randn(NN, HH, WW, CC).astype(np.float32)
    numBins = 1    # number of bins for block counter
    return NN, HH, WW, CC, RR, SS, UU, VV, BCH, BCW, numBins, mask, x


config = tf.ConfigProto(log_device_placement=False)
config.graph_options.optimizer_options.opt_level = -1


class TestSparseConvolutions(unittest.TestCase):
    def runTestForOneSize(self,
            layerSize,
            RS,
            UV,
            BOFFS=[-1, -2],
            checkGrads = False,
            sparsity = 70,
            do_add = False,
            N = 1,
            test_var = False,
            avgpool = False,
            use_atomics = False):

        tf.logging.set_verbosity(tf.logging.ERROR)
        print("============== Testing batch, layerSize = ", N, layerSize, "=========")
        results = [[], []]
        NN, HH, WW, CC, RR, SS, UU, VV, BCH, BCW, numBins, mask, x \
            = generateInputs(layerSize, RS, UV, BOFFS, N)

        # x doesn't need to be transposed, it's expected to be in NHWC
        offset = 10.0
        for transpose in [False, True]:
            for devStr in ["/gpu:0", "/cpu:0"]:
                tf.reset_default_graph()
                with tf.Session(config=config) as sess, tf.device(devStr):
                    if test_var:
                        x1000 = tf.Variable(x * 0.0 + offset, tf.float32)    # was: convert_to_tensor
                        sess.run(tf.global_variables_initializer())    # initialize x1000
                    #print(mask)
                    tol = np.percentile(x, sparsity)
                    if sparsity == 100:
                       tol = 1e32 # make sure at 100% sparsity we get zero blocks
                    a = tf.constant(mask, dtype=tf.float32)
                    print("-------------- BCNT=", BCH, BCW)
                    b = sbnet_module.reduce_mask(
                        a, tf.constant([BCH, BCW], dtype=tf.int32),
                        bsize=[RR, SS],
                        boffset=BOFFS,
                        bstride=[UU, VV],
                        tol=tol,
                        avgpool=avgpool)

                    # decouple indeterminstic portion into a separate subgraph for grad checker consistency
                    py_bin_counts, py_active_block_indices = sess.run(
                        [b.bin_counts, b.active_block_indices])

                    tf_bin_counts = tf.constant(py_bin_counts)
                    tf_active_block_indices = tf.constant(py_active_block_indices)
                    #bin_counts = tfx_print(b.bin_counts, "bin_counts")

                    tf_x = tf.convert_to_tensor(x, tf.float32)
                    dt0 = sbnet_module.cuda_op_timer(timer_name="my_timer", is_start=True)
                    with tf.control_dependencies([dt0]):
                        tf_x = tf.identity(tf_x)
                    blockStack = sbnet_module.sparse_gather(
                        tf_x,
                        tf_bin_counts,
                        tf_active_block_indices,
                        bsize=[RR, SS],
                        boffset=BOFFS,
                        bstride=[UU, VV],
                        transpose=transpose)
                    if test_var:
                        y1000 = sbnet_module.sparse_scatter_var(
                            blockStack,
                            tf_bin_counts,
                            tf_active_block_indices,
                            x1000,    # base variable to copy to output and overwrite on top of
                            bsize=[RR, SS],
                            boffset=BOFFS,
                            bstride=[UU, VV],
                            add=do_add,
                            atomic=use_atomics,
                            transpose=transpose)
                    else:
                        x1000 = tf.convert_to_tensor(x * 0.0 + offset, tf.float32)
                        y1000 = sbnet_module.sparse_scatter(
                            blockStack,
                            tf_bin_counts,
                            tf_active_block_indices,
                            x1000,    # base tensor to copy to output and overwrite on top of
                            bsize=[RR, SS],
                            boffset=BOFFS,
                            bstride=[UU, VV],
                            add=do_add,
                            atomic=use_atomics,
                            transpose=transpose)
                    dt = sbnet_module.cuda_op_timer(timer_name="my_timer", is_start=False)
                    with tf.control_dependencies([dt]):
                        y1000 = tf.identity(y1000)

                    result = sess.run([b, blockStack, y1000, dt])
                    if result[3] != -1.0:
                        print("CUDA time=", result[3])
                    #print("BLOCKS=", result[1])
                    #print("BLKIDS=", result[0])

                    result[0] = lambda: 0
                    result[0].bin_counts = py_bin_counts
                    result[0].active_block_indices = py_active_block_indices
                    result.append(x)
                    result.append(mask)

                    tidx = 1 if transpose else 0
                    results[tidx].append(result)

                    if checkGrads and not test_var:
                        blockStackResult = result[1]
                        err = gradient_checker.compute_gradient_error(
                            tf_x, x.shape, blockStack, blockStackResult.shape, x_init_value=x)
                        print("Device, grad error=", devStr, err)
                        self.assertTrue(err < 0.001)
                        #grads = gradients_impl.gradients([blockStack], [tf_x])
                        #resultGrads = sess.run([grads])
                        #print(resultGrads)

                        # if forward pass scatters are overlapping, some values will be overwritten indeterministically
                        # so gradient currently doesn't make sense without atomicAdd in forward scatter
                        if UU >= RR and VV >= SS:
                            y1000Result = result[2]
                            err = gradient_checker.compute_gradient_error(
                                [tf_x, x1000], [x.shape, x.shape],
                                y1000,
                                y1000Result.shape,
                                x_init_value=[x, x * 0.0 + offset])
                            self.assertTrue(err < 0.001)
                            print("Device, grad error=", devStr, err)

        # tidx = 0 : untransposed results
        # tidx = 1 : transposed results
        for tidx in range(2):
            icpu = 1
            igpu = 0
            rt = results[tidx]
            # check the input matched
            self.assertTrue(np.array_equal(rt[icpu][4], rt[igpu][4]))
            # check the mask matched
            self.assertTrue(np.array_equal(rt[icpu][5], rt[igpu][5]))

            reducedCpu = rt[icpu][0]
            reducedGpu = rt[igpu][0]
            cpuIndices = reducedCpu.active_block_indices[0:reducedCpu.bin_counts[0]]
            gpuIndices = reducedGpu.active_block_indices[0:reducedGpu.bin_counts[0]]
            self.assertTrue(reducedCpu.bin_counts[0] == reducedGpu.bin_counts[0]
                    )    # make sure the count of indices matches
            if sparsity == 100:
                self.assertTrue(
                    reducedCpu.bin_counts[0] == 1 and reducedCpu.active_block_indices[0] == 0)
            bin_count = reducedCpu.bin_counts[0]
            set0 = set([x for x in cpuIndices])
            set1 = set([x for x in gpuIndices])
            sorted = reducedGpu.active_block_indices[0:bin_count].argsort()
            self.assertTrue(set0 == set1)    # make sure the sets of indices match
            self.assertTrue(np.array_equal(cpuIndices, gpuIndices[sorted]))    # make sure sorted indices match

            gatheredCpu = rt[icpu][1]
            gatheredGpu = rt[igpu][1][sorted]
            if tidx == 0:
                gatheredCpuUT = gatheredCpu
                gatheredGpuUT = gatheredGpu
            else:
                gatheredCpuT = gatheredCpu
                gatheredGpuT = gatheredGpu
                # transposed is NCHW, convert to NHWC via [0, 2, 3, 1]
                self.assertTrue(np.array_equal(gatheredCpuUT, np.transpose(gatheredCpuT, [0, 2, 3, 1])))
                self.assertTrue(np.array_equal(gatheredCpuUT, np.transpose(gatheredGpuT, [0, 2, 3, 1])))
            gatherEq = np.array_equal(gatheredCpu, gatheredGpu)
            self.assertTrue(gatherEq)

            # check that scattered results match
            scatteredCpu = rt[icpu][2]
            scatteredGpu = rt[igpu][2]
            self.assertTrue(np.array_equal(scatteredCpu, scatteredGpu))
            #errIndex = np.unravel_index(np.absolute(gatheredCpu - gatheredGpu).argmax(), gatheredCpu.shape)
            #print(errIndex)

        self.assertTrue(np.array_equal(gatheredCpuT, np.transpose(gatheredCpuUT, [0, 3, 1, 2])))
        self.assertTrue(np.array_equal(gatheredGpuT, np.transpose(gatheredGpuUT, [0, 3, 1, 2])))
        print("========================= Test PASSED ==========================")

    def testSimple(self):
        self.runTestForOneSize((2, 3, 2), RS=(2, 3), UV=(3, 4))
        self.runTestForOneSize((5, 5, 1), RS=(3, 3), UV=(1, 1))
        self.runTestForOneSize((5, 5, 1), RS=(3, 3), UV=(1, 1))

    def testZeroBlockCount(self):
        # check 0-block-count from reduceMask
        self.runTestForOneSize((2, 2, 1), RS=(1, 1), UV=(1, 1),
                          BOFFS=(0, 0), checkGrads=True, sparsity=100)

    def testBasicGradients(self):
        # check basic grads
        self.runTestForOneSize((2, 2, 1), RS=(1, 1), UV=(1, 1),
                          BOFFS=(0, 0), checkGrads=True, sparsity=0)
        # multibatch
        self.runTestForOneSize((2, 2, 1), RS=(1, 1), UV=(1, 1),
                          BOFFS=(0, 0), checkGrads=True, sparsity=0, N=3)
        # slightly bigger tensor
        self.runTestForOneSize((6, 6, 1), RS=(1, 1), UV=(1, 1), checkGrads=True)

    def testOverlappingGradients(self):
        # check overlapping grads - exercise atomic reduce path
        self.runTestForOneSize((3, 3, 1), RS=(2, 2), UV=(1, 1),
                          BOFFS=(0, 0), checkGrads=True, sparsity=0)
        self.runTestForOneSize((13, 12, 1), RS=(2, 3), UV=(1, 2),
                          BOFFS=(0, 0), checkGrads=True, sparsity=0.1)

    def testSparsities(self):
        for sparsity in [10, 20, 30, 40, 50, 70, 90, 99]:
            self.runTestForOneSize((200, 300, 32), RS=(3, 4), UV=(2, 3),
                              sparsity=sparsity, avgpool=True, use_atomics=True)

    def testAvgPoolingWithHoles(self):
        self.runTestForOneSize((2, 3, 2), RS=(2, 3), UV=(3, 4), avgpool=True)
        self.runTestForOneSize((2, 3, 2), RS=(2, 3), UV=(3, 4), avgpool=False)

    def testChannelSweep(self):
        for cc in [15, 16, 24, 32, 48, 64]:
            self.runTestForOneSize((380, 480, cc), RS=(18, 18), UV=(18, 18))
        self.runTestForOneSize((200, 300, 32), RS=(16, 16), UV=(16, 16))

    def testSimplePermutations(self):
        for nn in [1, 7]:
            for tv in [True, False]:
                for avgpool in [True, False]:
                    for sp in [0, 10, 50, 99]:
                        args = {
                            "RS": (1, 1),
                            "UV": (1, 1),
                            "BOFFS": [0, 0],
                            "checkGrads": False,
                            "sparsity": sp,
                            "N": nn,
                            "test_var": tv,
                            "avgpool": avgpool
                        }
                        self.runTestForOneSize((1, 1, 1), **args)
                        self.runTestForOneSize((1, 1, 2), **args)
                        self.runTestForOneSize((1, 2, 3), **args)
                        self.runTestForOneSize((2, 1, 3), **args)
                        self.runTestForOneSize((1, 1, 1), **args)
                        self.runTestForOneSize((3, 2, 1), **args)
                        self.runTestForOneSize((1, 1, 1), **args)
                        self.runTestForOneSize((1, 2, 1), **args)
                        self.runTestForOneSize((3, 2, 1), **args)
                        self.runTestForOneSize((3, 2, 7), **args)
                        self.runTestForOneSize((3, 2, 7), **args)
                        self.runTestForOneSize((1, 1, 1024 * 49), **args)

    def testUnevenKernelSizes(self):
        self.runTestForOneSize((8, 10, 7), RS=(3, 4), UV=(3, 4))

    def testGapsSingleChannel(self):
        self.runTestForOneSize((8, 10, 1), RS=(3, 4), UV=(4, 5))

    def testGapsMultiChannel(self):
        self.runTestForOneSize((8, 10, 7), RS=(3, 4), UV=(4, 5))

    def testUnevenBoffs(self):
        self.runTestForOneSize((5, 5, 1), RS=(3, 2), UV=(3, 2), BOFFS=[-1, -2], checkGrads=True)

    def testUnevenNonOverlapping(self):
        self.runTestForOneSize((5, 3, 1), RS=(2, 3), UV=(2, 3), BOFFS=[0, 0])
        self.runTestForOneSize((5, 3, 1), RS=(2, 3), UV=(2, 3), BOFFS=[-1, -2])

    def testSimpleEvenOverlapping(self):
        # test even overlapping block sizes and strides
        self.runTestForOneSize((3, 3, 1), RS=(2, 2), UV=(1, 1), BOFFS=[0, 0])

    def testSimpleStidesWithHoles(self):
        # test block sizes/strides with holes
        self.runTestForOneSize((3, 3, 1), RS=(1, 1), UV=(2, 2))

    def testSimpleStridesWithHolesMultichannel(self):
        # test uneven block sizes/strides with holes + larger channel count
        self.runTestForOneSize((8, 10, 7), RS=(3, 4), UV=(4, 5))

    def benchmarkLayerSizes(self):
        layerSizes = ((300, 480, 64), (600, 960, 64), (300, 480, 256), (300, 480, 64), (300, 480, 256))
        for ls in layerSizes:
            self.runTestForOneSize(ls, RS=(18, 18), UV=(18, 18))

    def testLargeBlockSizes(self):
        # test very large block sizes
        self.runTestForOneSize((2048, 2048, 1), RS=(1500, 490), UV=(300, 390), use_atomics=True)

if __name__ == '__main__':
    unittest.main()
