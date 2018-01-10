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
# Benchmark performance for a synthetic top left corner solid mask.
#
# Usage:
# python benchmark_topleft.py --test [conv | res] --arch [resnet-50 | resnet-v2]
#
# Flags:
# --test: Which benchmark, a convolutional layer or a residual block.
# --arch: Which architecture, original ResNet-50  (high channel) or modified ResNet-v2 (low channel).
#
from __future__ import division, print_function

import os
import tensorflow as tf
import time

from argparse import ArgumentParser
from collections import namedtuple

from benchmark_configs import INPUT_SIZE_DICT, SPARSITY_LIST
from benchmark_utils import append_result, create_result, get_out_filename
from sparse_conv_perf import run_one, TestConfig, generate_top_left_mask

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 1
TOL = 0.05
AVGPOOL = True    # Use average pooling.
BLOCK_SIZE_LIST_LIST = [
    range(11, 33, 2),
    range(7, 25, 2),
    range(5, 21, 2),
    range(5, 21, 2),
]

perf_result_fields = [
    'H', 'W', 'C', 'K', 'BH', 'BW', 'sparsity', 'block_sparsity', 'dense_time', 'sparse_time',
    'speedup'
]

PerfResult = namedtuple('PerfResult', perf_result_fields)


def main():
    # Output performance metrics to a CSV file.
    res_block = args.test == 'res'
    out_file = get_out_filename(prefix='ours_topleft_{}'.format(args.test))
    print('Writing output to {}'.format(out_file))
    create_result(out_file, perf_result_fields)

    for xsize, bsize_list in zip(INPUT_SIZE_LIST, BLOCK_SIZE_LIST_LIST):
        # Benchmark dense convolution.
        if args.test == 'conv':
            xsize_ = [BATCH_SIZE] + [xsize[0], xsize[1], xsize[3]]
            ksize_ = [3, 3, xsize[3], xsize[3]]
        elif args.test == 'res':
            xsize_ = [BATCH_SIZE] + xsize[:-1]
            ksize_ = [3, 3, xsize[2], xsize[3]]
        test_config = TestConfig(
            xsize=xsize_,
            ksize=ksize_,
            strides=[1, 1, 1, 1],
            padding='SAME',
            bsize=None,
            is_sparse=False,
            tol=None,
            avgpool=None)
        with tf.Graph().as_default(), tf.Session() as sess:
            test_result = run_one(sess, None, test_config, res_block=res_block)
        dense_time = test_result.avg_time

        # Benchmark sparse convolution.
        for sparsity in SPARSITY_LIST:
            best_speedup = 0.0
            best_time = 0.0
            best_bsize = None
            mask = generate_top_left_mask(test_config.xsize, sparsity)
            for bsize in bsize_list:
                # For this synthetic mask, I found that rectangular block size does not help.
                bsize_h = bsize
                bsize_w = bsize
                # Whether the block size is larger than input size.
                if bsize_h > xsize[0] or bsize_w > xsize[1]:
                    continue
                test_config = TestConfig(
                    xsize=xsize_,
                    ksize=ksize_,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    bsize=[1, bsize_h, bsize_w, 1],
                    is_sparse=True,
                    tol=TOL,
                    avgpool=AVGPOOL)
                with tf.Graph().as_default(), tf.Session() as sess:
                    test_result = run_one(sess, mask, test_config, res_block=res_block)
                speedup = dense_time / test_result.avg_time
                if speedup > best_speedup:
                    best_bsize = (bsize_h, bsize_w)
                    best_speedup = speedup
                    best_time = test_result.avg_time
                    best_block_sparsity = test_result.block_sparsity
            result = PerfResult(
                H=xsize[0],
                W=xsize[1],
                C=xsize[2],
                K=xsize[3],
                BH=best_bsize[0],
                BW=best_bsize[1],
                sparsity=sparsity,
                block_sparsity=best_block_sparsity,
                dense_time=dense_time,
                sparse_time=best_time,
                speedup=best_speedup)
            append_result(out_file, result)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Sparse block\'s convolution and resnet blocks benchmark top left mask')
    parser.add_argument('--test', type=str, default='conv', choices=set(('conv', 'res')))
    parser.add_argument(
        '--arch', type=str, default='resnet-v2', choices=set(('resnet-50', 'resnet-v2')))
    args = parser.parse_args()
    print("Benchmarking with --test=%s --arch=%s" % (args.test, args.arch))
    INPUT_SIZE_LIST = INPUT_SIZE_DICT[args.arch]
    main()
