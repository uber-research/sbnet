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
# Benchmark submanifold top left mask performance.
#
# Usage:
# python benchmark_topleft.py --test [conv | res] --arch [resnet-50 | resnet-v2]
#
# Flags:
# --test: Which benchmark, a convolutional layer or a residual block.
# --arch: Which architecture, original ResNet-50  (high channel) or modified ResNet-v2 (low channel).
#
from __future__ import division, print_function

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from argparse import ArgumentParser
from collections import namedtuple

from perf import run_dense, run_sparse, generate_top_left_mask

# Import from our benchmark folder.
from benchmark_configs import INPUT_SIZE_DICT, SPARSITY_LIST
from benchmark_utils import append_result, create_result, get_out_filename

N_RUN_CONV = 15
N_RUN_RES = 15

perf_result_fields = ['H', 'W', 'C', 'K', 'sparsity', 'dense_time', 'sparse_time', 'speedup']
PerfResult = namedtuple('PerfResult', perf_result_fields)


def main():
    out_file = get_out_filename(prefix='submanifold_topleft_{}'.format(args.test))
    print('Writing output to {}'.format(out_file))
    create_result(out_file, perf_result_fields)
    for sz in INPUT_SIZE_LIST:
        for sparsity in SPARSITY_LIST:
            if args.test == 'conv':
                x = generate_top_left_mask([1, sz[3], sz[0], sz[1]], sparsity)
            elif args.test == 'res':
                x = generate_top_left_mask([1, sz[2], sz[0], sz[1]], sparsity)
            img_tensor = torch.FloatTensor(x)
            stream = torch.cuda.current_stream()
            nchw = img_tensor.size()
            if args.test == "conv":
                n_run = N_RUN_CONV
                res_block = False
            else:
                n_run = N_RUN_RES
                res_block = True
            dense_ms = run_dense(
                img_tensor, sz[3], res_block=res_block, n_warmup=n_run, n_run=n_run)
            sparse_ms = run_sparse(
                img_tensor, sz[3], res_block=res_block, n_warmup=n_run, n_run=n_run)
            result = PerfResult(
                H=sz[0],
                W=sz[1],
                C=sz[2],
                K=sz[3],
                sparsity=sparsity,
                dense_time=dense_ms,
                sparse_time=sparse_ms,
                speedup=dense_ms / sparse_ms)
            append_result(out_file, result)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Submanifold convolution and resnet blocks benchmarking script')
    parser.add_argument('--test', type=str, default='conv', choices=set(('conv', 'res')))
    parser.add_argument(
        '--arch', type=str, default='resnet-v2', choices=set(('resnet-50', 'resnet-v2')))
    args = parser.parse_args()
    print('Benchmarking with --test=%s --arch=%s' % (args.test, args.arch))
    INPUT_SIZE_LIST = INPUT_SIZE_DICT[args.arch]
    main()
