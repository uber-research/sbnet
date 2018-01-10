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

#ifndef KERNEL_SPARSE_GATHER_H
#define KERNEL_SPARSE_GATHER_H

template <typename Device, typename T> struct SparseGatherFunctor {
    void operator()(
        const Device& d,    // Device.
        const T* x,         // Input tensor.
        int N,              // Batch size of x.
        int H,              // Height of x.
        int W,              // Width of x.
        int C,              // Number of channels in x.
        T* y,               // Output tensor of shape=(numActive, bSzH, bSzW, C).
        int bOffsH0,        // Block padding offset height, negative.
        int bOffsW0,        // Block padding offset width, negative.
        int bSzH,           // Block size height.
        int bSzW,           // Block size width,
        int bStrH,          // Block stride, height.
        int bStrW,          // Block stride, width.
        int numActive, // Number of active blocks.
        const int64_t* activeBlockIndices, // Indices of active blocks.
        bool transpose
    );
};


template <typename Device, typename T> struct SparseScatterFunctor {
    void operator()(
        const Device& d,    // Device.
        const T* x,         // Input tensor of shape=(numActive, bSzH, bSzW, C).
        int N,
        int H,
        int W,
        int C,
        T* y,               // Output tensor of shape NHWC
        int bOffsH0,        // Block padding offset height, negative.
        int bOffsW0,        // Block padding offset width, negative.
        int bSzH,           // Block size height.
        int bSzW,           // Block size width,
        int bStrH,          // Block stride, height.
        int bStrW,          // Block stride, width.
        int numActive,      // Number of active blocks.
        const int64_t* activeBlockIndices, // Indices of active blocks.
        bool add,
        bool transpose,
        bool atomic
    );
};


template<typename Device, typename T> struct CopyTensorFunctor {
    void operator()(
        const Device& d,
        T* dst,
        const T* src,
        int count);
    cudaStream_t* getStream(const Device& d);
};

#endif // KERNEL_SPARSE_GATHER_H

