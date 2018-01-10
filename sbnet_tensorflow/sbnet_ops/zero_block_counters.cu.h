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

#ifndef ZERO_BLOCK_COUNTERS_H
#define ZERO_BLOCK_COUNTERS_H

#include "cuda_helpers.h"

// Zeroes out the binCounters for the subsequent call to reduceMask
__device__ void zeroBlockCounters_t(unsigned int numBins, unsigned int* binCounts)
{
    // initialize binCounts to zero
    // how many blocks of threads to cover all bins?
    int numLoops = DIVUP(numBins, gridDim.x);
    for (int iBlockLoop = 0; iBlockLoop < numLoops; iBlockLoop++)
    {
        int writeIdx = (blockIdx.x*blockDim.x + threadIdx.x);
        writeIdx += iBlockLoop*gridDim.x*blockDim.x;
        if (writeIdx < numBins)
        {
            binCounts[writeIdx] = 0;
        }
    }
}

extern "C" {
// kernel entry point
__global__ void zeroBlockCounters(
    unsigned int numBins,
    unsigned int* binCounts
)
{
    dprintBlockGrid();
    zeroBlockCounters_t(numBins, binCounts);
}
}

#endif
