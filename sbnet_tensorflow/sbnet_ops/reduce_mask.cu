/*

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

*/

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "reduce_mask.h"
#include "zero_block_counters.cu.h"
#include "reduce_mask.cu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "cuda_helpers.h"

using namespace tensorflow;
using std::cout;
using std::endl;

// Define the GPU implementation that launches the CUDA kernel.
template <typename T> struct ReduceMaskFunctor<GPUDevice, T> {
    void operator()(const GPUDevice& d, // Device.
        const T* mask,                  // Mask array.
        int N,                          // Batch dimension of the mask.
        int H,                          // Height of the mask.
        int W,                          // Width of the mask.
        float threshold,                // Threshold for being active.
        int bOffsH0,                    // Block padding offset height, negative.
        int bOffsW0,                    // Block padding offset width, negative.
        int bSzH,                       // Block size height.
        int bSzW,                       // Block size width.
        int bStrH,                      // Block stride, height.
        int bStrW,                      // Block stride, width.
        int bCntH,                      // Number of blocks, height.
        int bCntW,                      // Number of blocks, width.
        unsigned int numBins,
        unsigned int binSize,
        long long* activeBlockIndices,  // Indices of active blocks.
        int* binCounts,                 // Number of indices of active blocks.
        bool avgPool
        )
    {
        zeroBlockCounters<<<24, 1024, 0, d.stream()>>>(numBins, (unsigned int*) binCounts);
        dim3 block(std::min(DIVUP(bSzH*bSzW, 32)*32, 1024), 1, 1);
        dim3 grid(bCntW, bCntH, N);
        reduceMask<<<grid, block, 0, d.stream()>>>(mask, N, H, W, // C is assumed to be 1
            threshold, // value to consider non-sparse block
            numBins,   // number of bins to partition activeBlockIndices to reduce atomics pressure
            binSize,
            (unsigned int*) binCounts, // counts for sub-blocks, initialized to 0
            (unsigned long long*) activeBlockIndices,
            bOffsH0,
            bOffsW0,      // generally negative - first block element offset for correct padding
            bSzH, bSzW,   // block sizes
            bStrH, bStrW, // block strides
            bCntH, bCntW, // block counts
            avgPool
            );
    }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct ReduceMaskFunctor<GPUDevice, float>;

#endif // GOOGLE_CUDA
