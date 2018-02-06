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

#ifndef KERNEL_REDUCE_MASK_H
#define KERNEL_REDUCE_MASK_H

template <typename Device, typename T> struct ReduceMaskFunctor {
    void operator()(const Device& d,   // Device.
        const T* mask,                 // Mask array.
        int N,                         // Batch dimension of the mask.
        int H,                         // Height of the mask.
        int W,                         // Width of the mask.
        float threshold,               // Threshold for being active.
        int bOffsH0, // Block padding offset height, negative.
        int bOffsW0, // Block padding offset width, negative.
        int bSzH,    // Block size height.
        int bSzW,    // Block size width,
        int bStrH,   // Block stride, height.
        int bStrW,   // Block stride, width.
        int bCntH,   // Number of blocks, height.
        int bCntW,   // Number of blocks, width.
        unsigned int numBins,
        unsigned int binSize,
        long long* activeBlockIndices, // Indices of active blocks.
        int* binCounts,
        bool avgPool
        );
};

#endif // KERNEL_REDUCE_MASK_H
