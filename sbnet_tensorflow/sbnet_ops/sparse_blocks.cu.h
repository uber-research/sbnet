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

#ifndef SPARSE_BLOCKS_H
#define SPARSE_BLOCKS_H

#include "cuda_helpers.h"
#include "device_atomic_functions.h"

using namespace ST;

struct TileShared {
    float * tile_;
    int rr_, ss_, cc_;
    __forceinline__ __device__ void init(float* ext_shmem, int rr, int ss, int cc)
    {
        tile_ = ext_shmem;
        rr_ = rr;
        ss_ = ss;
        cc_ = cc;
    }
    __forceinline__ __device__ float& val(int hh, int ww, int cc) {
        constexpr int O = 0;
        return tile_[(hh)*ss_*(cc_+O) + (ww)*(cc_+O) + cc];
    }
};

template<int RR1, int SS, int CC1>
struct TileLocal {
    __forceinline__ __device__ void init(float* ext_shmem, int rr, int ss, int cc)
    {
        assert(rr == RR1);
        assert(ss == SS);
        assert(cc == CC1);
    }

    float tile_[RR1][SS][CC1+1];
    __forceinline__ __device__ float& val(int hh, int ww, int cc) {
        return tile_[hh][ww][cc];
    }
};

// Work partition strategy:
// Reads are coalesced in C, writes are coalesced in (HW)
// Load into tiles of size BhBwCtile
// Wrap threads around BhBwC
// CUDA blocks are mapped to tiles (K CUDA blocks per sparsity block)
// (Currently 1 to 1 to indices in activeBlockIndices)
// (TODO: use all binCounts if needed for perf)
// tC and tH are further tiled using tC1 and TH1b-long
// tiles to create smaller chunks of work per block
template<int tNTHREADS, typename tHb, typename tH1b, typename tWb, typename tC, typename tC1, typename tTranspose, typename tTile>
__device__ void blockGatherTiled_t(
    const float* x,                        // NHWC
    // currently only supporting single bin
    const u64* activeBlockIndices,
    // variable size output but conservative worst-case memory alloc (0% sparsity, all blocks active)
    float* y,
    const int N, const int H, const int W, // C=tC is templated
    const int bOffsH0, const int bOffsW0,  // generally negative - first block element offset for correct padding
    const int bStrH, const int bStrW,      // block strides
    tHb tHbArg, tH1b tH1bArg, tWb tWbArg, tC tCArg, tC1 tC1Arg, tTranspose tTransposeArg
    )
{
    // RS in this kernel refers to tHb, tWb
    const int R = tHbArg.get(), R1 = tH1bArg.get(), S = tWbArg.get(), C1 = tC1Arg.get();
    const int RS = R*S, R1S = R1*S;
    const int transpose = tTransposeArg.get();
    const int C = tCArg.get();
    const int SC = S*C, RSC = R*S*C;
    const int WC = W*C;
    assert(R1 <= R);
    assert(C1 <= C);

    int biN, biH, biW; // retrieve the block index from sparse u64 index
    const u64 sbi = activeBlockIndices[blockIdx.x];
    const int c1i = blockIdx.y;
    const int r0 = blockIdx.z*R1;
    SparseBlockIndex::from64Bit(sbi, biN, biH, biW);

    // wrap threads around HbWbK
    int bh0 = bOffsH0 + bStrH*biH; // current block's first pixel hw
    int bw0 = bOffsW0 + bStrW*biW;
    int bn0Offs = biN*H*W*C;

    extern __shared__ float shmem[];
    __shared__ tTile tile;
    tile.init(shmem, R1, S, C1);

#pragma unroll 4
    // read the tile
    // logical index inside block
    for (int r1scStrided = 0; r1scStrided < R1S*C1; r1scStrided += tNTHREADS)
    {
        int r1sc = r1scStrided + threadIdx.x;
        // unwrap R1SC1 -> intraBh, intraBw, c
        unsigned tileC = r1sc % C1;
        unsigned intraBw = (r1sc/C1) % S;
        unsigned tileH = (r1sc/C1) / S;
        if (tileH >= R1) // can roll over if nthreads doesn't align
            continue;
        // map intra-block coords to whole tensor now
        unsigned w0 = bw0 + intraBw;
        unsigned h0 = bh0 + tileH + r0;
        unsigned c = tileC + C1*c1i;
        int offsRead = bn0Offs + h0*WC + w0*C + c;
        float readVal = 0.0f;
        if (w0 < W && h0 < H && c < C)
            readVal = __ldg(x + offsRead);
        tile.val(tileH, intraBw, tileC) = readVal;
    }

    // flush the tile shmem
    __syncthreads();

    // write the tile
#pragma unroll 4
    for (int cr1sStrided = 0; cr1sStrided < R1S*C1; cr1sStrided += tNTHREADS)
    {
        int cr1s = cr1sStrided + threadIdx.x;
            // unwrap CRS -> c, intraBh, intraBw
            unsigned tileH, intraBw, tileC;
            if (transpose) {
                intraBw = cr1s % S;
                tileH = (cr1s/S) % R1;
                tileC = cr1s / R1S;
                if (tileC >= C1)
                    continue;
            } else {
                tileC = cr1s % C1;
                intraBw = (cr1s/C1) % S;
                tileH = (cr1s/C1) / S;
                if (tileH >= R1)
                    continue;
            }
            int c = tileC + C1*c1i;
            if (c >= C) // tail C1 block can have c >= C
                continue;
            // map intra-block coords to whole tensor now

            float readVal = tile.val(tileH, intraBw, tileC);
            unsigned intraBh = tileH + r0;
            if (intraBh >= R)
                continue;
            int offsWrite;
            if (transpose)
                // NCHW offset
                offsWrite = blockIdx.x*RSC + c*RS + intraBh*S + intraBw;
            else
                // NHWC offset
                offsWrite = blockIdx.x*RSC + intraBh*SC + intraBw*C + c;

            y[offsWrite] = readVal;
    }
}

template<int tNTHREADS, typename tHb, typename tH1b, typename tWb, typename tC, typename tC1,
    typename tAdd, typename tTranspose, typename tAtomic, typename tTile>
__device__ void blockScatterTiled_t(
    const float* x,                        // blockDim.x*tHb*tWb*tC - input tensor
    // currently only supporting single bin
    const u64* activeBlockIndices, // same indices as for blockGather
    // variable size output but conservative worst-case memory alloc (0% sparsity, all blocks active)
    float* y,                              // NHWC
    const int N, const int H, const int W, // C=tC is templated
    const int bOffsH0, const int bOffsW0,  // generally negative - first block element offset for correct padding
    const int bStrH, const int bStrW,      // block strides
    tHb tHbArg, tH1b tH1bArg, tWb tWbArg, tC tCArg, tC1 tC1Arg,
    tAdd tAddArg, tTranspose tTransposeArg, tAtomic tAtomicArg)
{
    // RS in this kernel refers to tHb, tWb
    const int R = tHbArg.get(), R1 = tH1bArg.get(), S = tWbArg.get(), C1 = tC1Arg.get();
    const int RS = R*S, R1S = R1*S;
    const int add = tAddArg.get(), transpose = tTransposeArg.get();
    const int atomic = tAtomicArg.get();

    const int C = tCArg.get();
    const int SC = S*C, RSC = R*S*C;
    assert(R1 <= R);
    assert(C1 <= C);

    int biN, biH, biW; // retrieve the block index from sparse u64 index
    const u64 sbi = activeBlockIndices[blockIdx.x];
    const int c0 = blockIdx.y*C1;
    const int r0 = blockIdx.z*R1;
    SparseBlockIndex::from64Bit(sbi, biN, biH, biW);


    // wrap threads around HbWbK
    int bh0 = bOffsH0 + bStrH*biH; // current block's first pixel hw
    int bw0 = bOffsW0 + bStrW*biW;
    int bn0Offs = biN*H*W*C;

    extern __shared__ float shmem[];
    __shared__ tTile tile;
    tile.init(shmem, R1, S, C1);

    // read the tile
#pragma unroll 4
    for (int c1r1sStrided = 0; c1r1sStrided < R1S*C1; c1r1sStrided += tNTHREADS)
    {
        int c1r1s = c1r1sStrided + threadIdx.x;
            // unwrap CRS -> (c, intraBh, intraBw) or CHW variant
            unsigned tileH, intraBw, tileC;
            if (transpose) {
                intraBw = c1r1s % S;
                tileH = (c1r1s/S) % R1;
                tileC = c1r1s / R1S;
                if (tileC >= C1)
                    continue;
            } else {
                tileC = c1r1s % C1;
                intraBw = (c1r1s/C1) % S;
                tileH = (c1r1s/C1) / S;
                if (tileH >= R1)
                    continue;
            }
            int c = tileC + c0;
            if (c >= C) // tail C1 block can have c >= C
                continue;
            // map intra-block coords to whole tensor now
            unsigned intraBh = tileH + r0;
            if (intraBh >= R)
                continue;

            int offsRead;
            if (transpose)
                // NCHW offset
                offsRead = blockIdx.x*RSC + c*RS + intraBh*S + intraBw;
            else
                offsRead = blockIdx.x*RSC + intraBh*SC + intraBw*C + c;
            // shmem tile is HWC
            tile.val(tileH, intraBw, tileC) = __ldg(x + offsRead);
    }

    // flush the tile shmem
    __syncthreads();

#pragma unroll 4
    // write the tile
    // logical index inside block
    for (int r1scStrided = 0; r1scStrided < R1S*C1; r1scStrided += tNTHREADS)
    {
        int r1sc = r1scStrided + threadIdx.x;
        unsigned tileC = r1sc % C1;
        unsigned intraBw = (r1sc/C1) % S;
        unsigned tileH = (r1sc/C1) / S;
        if (tileH >= R1) // can roll over if nthreads doesn't align
            continue;
        // map intra-block coords to whole tensor now
        unsigned intraBh = tileH + r0;
        if (intraBh >= R)
            continue;
        unsigned w0 = bw0 + intraBw;
        unsigned h0 = bh0 + intraBh;
        unsigned c = tileC + c0;
        // intraBh can go over R
        if (w0 < W && h0 < H && c < C)
        {
            float readVal = tile.val(tileH, intraBw, tileC);
            int offsWrite = bn0Offs + h0*W*C + w0*C + c;
            if (add) {
                if (atomic)
                    atomicAdd(y+offsWrite, readVal);
                else
                    y[offsWrite] += readVal;
            } else {
                y[offsWrite] = readVal;
            }
        }
    }
}

// versions of kernel entry points with constexpr-based parameters
template<int tNTHREADS, int RR, int RR1, int SS, int CC1, bool TR>
__global__ void blockGatherTiled0(
    const float* x,                               // NHWC
    // currently only supporting single bin
    const u64* activeBlockIndices, // result
    // variable size output but requires conservative worst-case memory alloc (0% sparsity, all blocks active)
    float* y,
    const int N, const int H, const int W, const int C, // dimensions for x
    const int bOffsH0, const int bOffsW0,         // generally negative - first block element offset for correct padding
    const int bStrH, const int bStrW              // block strides
)
{
    blockGatherTiled_t<tNTHREADS, getConst<RR>, getConst<RR1>, getConst<SS>, getVar, getConst<CC1>, getConst<TR>, TileShared>(
                    x, activeBlockIndices, y, N, H, W, bOffsH0, bOffsW0, bStrH, bStrW,
                    getConst<RR>(), getConst<RR1>(), getConst<SS>(), getVar(C), getConst<CC1>(), getConst<TR>());
}

template<int tNTHREADS, int RR, int RR1, int SS, int CC1, bool ADD, bool TR, bool ATOMIC>
__global__ void blockScatterTiled0(
    const float* x,                               // NHWC
    // currently only supporting single bin
    const u64* activeBlockIndices, // result
    // variable size output but requires conservative worst-case memory alloc (0% sparsity, all blocks active)
    float* y,
    const int N, const int H, const int W, const int C, // dimensions for x
    const int bOffsH0, const int bOffsW0,         // generally negative - first block element offset for correct padding
    const int bStrH, const int bStrW              // block strides
)
{
    blockScatterTiled_t<
        tNTHREADS, getConst<RR>, getConst<RR1>, getConst<SS>, getVar,
        getConst<CC1>, getConst<ADD>, getConst<TR>, getConst<ATOMIC>, TileShared>
    (
        x, activeBlockIndices, y, N, H, W, bOffsH0, bOffsW0, bStrH, bStrW,
        getConst<RR>(), getConst<RR1>(), getConst<SS>(), getVar(C), getConst<CC1>(),
        getConst<ADD>(), getConst<TR>(), getConst<ATOMIC>()
    );
}

// versions of kernel entry points with variable-based parameters
template<int tNTHREADS>
__global__ void blockGatherTiled1(
    const float* x,                               // NHWC
    const u64* activeBlockIndices, // result
    float* y,
    const int N, const int H, const int W, const int C, // dimensions for x
    const int bOffsH0, const int bOffsW0,         // generally negative - first block element offset for correct padding
    const int bStrH, const int bStrW,              // block strides
    const int RR, const int RR1, const int SS,
    const int CC1, const int TR
)
{
    blockGatherTiled_t<tNTHREADS, getVar, getVar, getVar, getVar, getVar, getVar, TileShared>(
        x, activeBlockIndices, y, N, H, W, bOffsH0, bOffsW0, bStrH, bStrW,
        getVar(RR), getVar(RR1), getVar(SS), getVar(C), getVar(CC1), getVar(TR));
}

template<int tNTHREADS>
__global__ void blockScatterTiled1(
    const float* x,                               // NHWC
    // currently only supporting single bin
    const u64* activeBlockIndices, // result
    // variable size output but requires conservative worst-case memory alloc (0% sparsity, all blocks active)
    float* y,
    const int N, const int H, const int W, const int C, // dimensions for x
    const int bOffsH0, const int bOffsW0,         // generally negative - first block element offset for correct padding
    const int bStrH, const int bStrW,              // block strides
    const int RR, const int RR1, const int SS,
    const int CC1, const int ADD, const int TR, const int ATOMIC
)
{
    blockScatterTiled_t<tNTHREADS, getVar, getVar, getVar, getVar, getVar, getVar, getVar, getVar, TileShared>(
        x, activeBlockIndices, y, N, H, W, bOffsH0, bOffsW0, bStrH, bStrW,
        getVar(RR), getVar(RR1), getVar(SS), getVar(C), getVar(CC1), getVar(ADD), getVar(TR), getVar(ATOMIC));
}


#endif
