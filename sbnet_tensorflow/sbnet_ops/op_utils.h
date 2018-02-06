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

#ifndef OP_UTILS_H
#define OP_UTILS_H

#include "tensorflow/core/util/padding.h"

// Fuses three uint16 into one int64.
static inline int64_t to64Bit(uint16_t x, uint16_t y, uint16_t z)
{
    return (int64_t(x) << 32) + (int64_t(y) << 16) + int64_t(z);
}

static inline void from64Bit(int64_t val, int& n, int& h, int& w)
{
    w = val & 0xFFFF;
    h = (val >> 16) & 0xFFFF;
    n = (val >> 32) & 0xFFFF;
}

#define DIVUP(a, b) ( ((a)+(b)-1)/(b) )

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort)
           exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#endif
