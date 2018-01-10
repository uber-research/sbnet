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

#ifndef REDUCE_MASK_H
#define REDUCE_MASK_H

#include "cuda_helpers.h"
#include "zero_block_counters.cu.h"
#include "reduce_mask.cu.h"

using namespace ST;

//
// Mask is NHW1
// This tensor can be quite small, say highest res is (64,1920,1200,1)
// Could be as small as (64,32,32,1) or even (64,7,7,1) for last ImageNet layer
//
// One possible work partition strategy assuming larger batch:
// wrap HW around tNTHREADS, run N blocks, one block per batch,
// reduce the count inside each block
// use atomicAdd to reduce the total number of blocks
// 
// For small batch inference it's going to be better to have a HW-blocked kernel.
// This works for, say, 1x1920x1200x1 block size 
// Sometimes it's going to be difficult to utilize the GPU.
// For instance how do we partition a 1x7x7 with block size 1?
// 
// Perhaps we can do N*bCntH*bCntW blocks and wrap the threads around block pixels?
// there's going to be some duplication in reads/BW waste but the inputs should be small anyway
// N*BCH*BCW blocks kernel: blockIdx.x=[0, N)
// tNTHREADS is tHb*tWb
//
// blockDim.x = tbH*tbW
// gridDim = (x=bCntW, y=bCntH, z=N)
// So basically run a CUDA block per sparsity block
// threadIdx.x = intra-block w+h*W, rounded up to 32 (warpLanes)
//
template<typename tbH, typename tbW>
__device__ void reduceMask_t(
    const int nActualThreads,                 // tbH*tbW rounded up to warpSize
    const float* mask, int N, int H, int W,   // C is assumed to be 1
    const float threshold,                    // value to consider non-sparse block
    //int* reducedMask,                       // space for resulting binary max>threshold mask per sparsity block
    unsigned int  numBins,                    // number of bins to partition activeBlockIndices to reduce atomics pressure
    unsigned int  binSize,
    unsigned int* binCounts,                  // counts for sub-blocks, initialized to 0
    u64* activeBlockIndices,                  // result
    const int bOffsH0, const int bOffsW0,     // generally negative - first block element offset for correct padding
    const int bStrH, const int bStrW,         // block strides
    const int bCntH, const int bCntW,         // block counts
    tbH tbHArg, tbW tbWArg, bool avgPooling)  // do maxpool if avgPooling is false
{
    const int bH = tbHArg.get(), bW = tbWArg.get();

    int blockW0 = bOffsW0 + bStrW*blockIdx.x;
    int blockH0 = bOffsH0 + bStrH*blockIdx.y;
    int n = blockIdx.z;
    // one thread per sparsity block pixel
    const int roundedUpThreads = DIVUP(bH*bW, warpLanes)*warpLanes;
    float mx = avgPooling ? 0.0f : -1e30;
    // allocate and initialize shmem for block reduce
    constexpr int maxBlockDim = 1024;
    assert(blockDim.x <= maxBlockDim);
    __shared__ float shmemx[maxBlockDim];
    for (int initOffs = 0; initOffs < maxBlockDim; initOffs += blockDim.x)
        if (initOffs + threadIdx.x < maxBlockDim)
            shmemx[initOffs+threadIdx.x] = avgPooling ? 0.0f : -1e30f;
    __syncthreads();

    // for large sparsity blocks we need multiple CUDA block loops
    for (int tOffs = 0; tOffs < roundedUpThreads; tOffs+=blockDim.x)
    {
        int tid = threadIdx.x + tOffs;
        const float* blockStartN = mask + n*H*W;
        float readVal = avgPooling ? 0.0f : -1e30f; // this value will be used to pad the warp
        if (tid < bH*bW) { // TODO: not needed?
            int woffs = tid % bW;
            int hoffs = tid / bW;
            unsigned bhh = hoffs+blockH0, bww = woffs + blockW0;
            if (bhh < H && bww < W)
                readVal = blockStartN[bhh*W + bww];
        }

        // actual number of threads is rounded up to 32 but padded with zeroes
        // warp reduce for all threads
        mx = avgPooling ? (mx + readVal) : max(mx, readVal);
        #pragma unroll
        for (int offset = warpLanes/2; offset > 0; offset /= 2) {
            float warped = __shfl_down(mx, offset);
            mx = avgPooling ? (mx + warped) : max(mx, warped);
        }

        // store (first elems from) warp reduces into shmem
        if (tid % warpLanes == 0) {
            int offs = tid/warpLanes; // tid includes tOffs
            int offsWrap = offs%blockDim.x;
            if (avgPooling)
                // atomics not needed here since we wrap around each blockDim.x
                shmemx[offsWrap] += mx;
            else
                shmemx[offsWrap] = max(shmemx[offsWrap], mx);
        }
        __syncthreads();
    } // tOffs

    // final reduce over all warps
    if (threadIdx.x == 0) {
        float mx1 = shmemx[0];
        // For sizes >= blockIdx.x we already reduced in the above loop
        const int numWarps = min(DIVUP(bH*bW, warpLanes), blockIdx.x);
        #pragma unroll
        for (int iWarp = 1; iWarp < numWarps; iWarp++)
            mx1 = avgPooling ? (mx1 + shmemx[iWarp]) : max(mx1, shmemx[iWarp]);

        if (avgPooling)
            mx1 /= float(bH*bW);

        if (mx1 > threshold) {
            // now we have the maximums computed for each block
            // we need to write out the maximums, total over-threshold count across grid
            // at this point the number of blocks is grid size, so N*bCntH*bCntW
            // bad case scenario is say 4*64*64 (larger batch won't fit into memory)
            // so we can have ~16k blocks
            // need an efficient gmem reduction
            unsigned int blockIndex = n*bH*bW + blockIdx.y*bW + blockIdx.x;
            unsigned int myBin = ((blockIndex*100017+1234567)>>4) % numBins;
            unsigned int inBinOffs;
            // check for bin overflow
            while ((inBinOffs = atomicAdd(&binCounts[myBin], unsigned(1))) >= binSize)
            {
                atomicSub(&binCounts[myBin], unsigned(1));
                myBin++;
            }
            u64 as64 = SparseBlockIndex::to64Bit(blockIdx.z, blockIdx.y, blockIdx.x);

            activeBlockIndices[myBin*binSize+inBinOffs] = as64;
        } // if (mx1 > threshold)
    } // if (tid == 0)
}

extern "C" {
// kernel entry point
__global__ void reduceMask(
    const float* mask, int N, int H, int W,   // C is assumed to be 1
    const float threshold,                    // value to consider non-sparse block
    unsigned int  numBins,                    // number of bins to partition activeBlockIndices to reduce atomics pressure
    unsigned int  binSize,
    unsigned int* binCounts,                  // counts for sub-blocks, initialized to 0
    u64* activeBlockIndices,                  // result: block indices split into bins, currently assuming even sized bins.
                                              // the counter will spill into the next bin on overflow.
                                              // It is expected that activeBlockIndices is allocated enough memory for worst case
                                              // number of active indices, ie all blocks are active
    const int bOffsH0, const int bOffsW0,     // generally negative - first block element offset for correct padding
    const int bSzH, const int bSzW,           // block sizes
    const int bStrH, const int bStrW,         // block strides
    const int bCntH, const int bCntW,         // block counts
    bool avgPool)
{
    const int roundUpThreads = DIVUP(bSzH*bSzW, warpLanes)*warpLanes;
    assert((roundUpThreads == blockDim.x || roundUpThreads > 1024) &&
        "Error in reduceMask_t: blockDim.x must be a multiple of warpLanes\n");
    assert(blockDim.y == 1 && blockDim.z == 1 &&
        "Expected block shape=(x=nThreads, y=1, z=1)");

    reduceMask_t<getVar, getVar>(
        roundUpThreads, mask, N, H, W,
        threshold, numBins, binSize, binCounts,
        activeBlockIndices, bOffsH0, bOffsW0,
        bStrH, bStrW, bCntH, bCntW, getVar(bSzH), getVar(bSzW), avgPool);
}

}

#endif
