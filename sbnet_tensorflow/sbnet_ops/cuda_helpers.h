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

#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <stdio.h>
#include "assert.h"

#define DIVUP(a, b) ( ((a)+(b)-1)/(b) )

namespace ST {
    typedef unsigned long long u64;
    typedef long long          i64;
    typedef unsigned int       u32;
    typedef int                i32;
    typedef unsigned short     u16;
    typedef short              i16;
    typedef unsigned char      u8;
    typedef char               i8;
}

// template facilities for reusing the same kernel with both constexpr and variable params
template<const int i>
struct getConst
{
    __device__ __inline__ constexpr int get() const { return i; }
};

struct getVar
{
    __device__ __inline__ constexpr int get() const { return i_; }
    __device__ __inline__ constexpr getVar(const int i) : i_(i) { }
    int i_;
};

namespace {
constexpr int warpLanes = 32;
}

#define CONST constexpr

__device__ struct XYZ {
    int x, y, z;
    __device__ XYZ(int ax = 0, int ay = 0, int az = 0) { x = ax; y = ay; z = az; };
};

__device__ struct SparseBlockIndex {
    static inline __device__ ST::u64 to64Bit(ST::u16 n, ST::u16 h, ST::u16 w) {
        return (ST::u64(n)<<32)+(ST::u64(h)<<16)+ST::u64(w);
    }
    static inline __device__ void from64Bit(ST::u64 val, int& n, int& h, int& w) {
        w = val & 0xFFFF;
        h = (val >> 16) & 0xFFFF;
        n = (val >> 32) & 0xFFFF;
    }
};


// Utility macros for debug prints
#if 0 // enable for debugging, otherwise it will compile out
#define dprintEQ(bid, tid, args) \
    if (blockIdx.x == bid.x && blockIdx.y == bid.y && blockIdx.z == bid.z && threadIdx.x == tid.x) \
    { printf args; }
#define dprintLE(bid, tid, args) \
    if (blockIdx.x <= bid.x && blockIdx.y <= bid.y && blockIdx.z <= bid.z && threadIdx.x <= tid.x) \
    { printf args; }
#define dprintBlockGrid() \
    dprintEQ(XYZ(), XYZ(), ("blockDim=(%d,%d,%d), gridDim=(%d,%d,%d)\n", \
        blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z))
#else
#define dprintEQ(bid, tid, args)
#define dprintLE(bid, tid, args)
#define dprintBlockGrid()
#endif

#define dprintBlockThreadIndices() \
    printf("threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n", \
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z)

#endif
