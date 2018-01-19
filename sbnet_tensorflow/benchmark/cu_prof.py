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

from .benchmark_utils import prefix_path
import ctypes

_cudart = ctypes.CDLL('libcudart.so')


def cu_prof_start():
    # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    # the return value will unconditionally be 0. This check is just in case it changes in
    # the future.
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)


def cu_prof_stop_func(func, do_trace=False):
    """Profile a single function with both CUDA and TF trace."""
    import tensorflow as tf
    from tensorflow.python.client import timeline
    if do_trace:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    else:
        run_options = tf.RunOptions()
    run_metadata = tf.RunMetadata()
    with tf.Graph().as_default(), tf.Session() as sess:
        func(sess, run_options, run_metadata)
    # Create the Timeline object, and write it to a json
    if do_trace:
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(prefix_path('timeline.json'), 'w') as f:
            f.write(ctf)
        print('Done writing timeline.')
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)
