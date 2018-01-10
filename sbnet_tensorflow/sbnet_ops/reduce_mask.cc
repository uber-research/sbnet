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

#define EIGEN_USE_THREADS
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"

#include "op_utils.h"
#include "reduce_mask.h"

using namespace tensorflow;
using std::cout;
using std::endl;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU implementation of reduce mask op.
// This is a naive CPU implementation, just for reference comparison/testing purposes.
template <typename T> struct ReduceMaskFunctor<CPUDevice, T> {
    void operator()(const CPUDevice& d, // Device.
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
        int64* activeBlockIndices,      // Indices of active blocks.
        int* binCounts,                 // Number of active indices.
        bool avgPool
        )
    {
        int count = 0;
        assert(numBins == 1);
        const int C = 1;
        for (int n = 0; n < N; ++n) {
        for (int bh = 0; bh < bCntH; ++bh) {
        for (int bw = 0; bw < bCntW; ++bw) {
            int h0 = bOffsH0 + bh * bStrH;
            int w0 = bOffsW0 + bw * bStrW;
            bool active = false; // Whether a block is active.
            float sum = 0.0f;
            for (int hh = std::max(0, h0); hh < h0 + bSzH && hh < H; ++hh) {
            for (int ww = std::max(0, w0); ww < w0 + bSzW && ww < W; ++ww) {
            for (int cc = 0; cc < C; cc++) {
                float val = mask[n*H*W*C + hh*W*C + ww*C + cc];
                if (avgPool)
                    sum += val;
                else
                    active |= (val > threshold);
            } } }
            if (avgPool)
                active = ( (sum/(bSzH*bSzW)) > threshold );
            if (active)
                activeBlockIndices[count++] = to64Bit((uint16)n, (uint16)bh, (uint16)bw);
        } } }
        binCounts[0] = count;
    }
};

REGISTER_OP("ReduceMask")
    .Attr("T: {float}")
    .Attr("bsize: list(int)")
    .Attr("bstride: list(int)")
    .Attr("boffset: list(int)")
    .Attr("tol: float")
    .Attr("avgpool: bool = false")
    .Input("mask: T")
    .Input("dynamic_bcount: int32")
    .Output("active_block_indices: int64")
    .Output("bin_counts: int32");

// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T> class ReduceMaskOp : public OpKernel {
public:
    explicit ReduceMaskOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
        std::vector<int> bsize, bstride, boffset;
        OP_REQUIRES_OK(context, context->GetAttr("bsize", &bsize));
        OP_REQUIRES_OK(context, context->GetAttr("bstride", &bstride));
        OP_REQUIRES_OK(context, context->GetAttr("boffset", &boffset));
        OP_REQUIRES_OK(context, context->GetAttr("avgpool", &avgpool_));
        bSzH_    = bsize[0];   bSzW_ = bsize[1];
        bStrH_   = bstride[0]; bStrW_ = bstride[1];
        bOffsH0_ = boffset[0]; bOffsW0_ = boffset[1];
        OP_REQUIRES_OK(context, context->GetAttr("tol", &tol_));
    }

    void Compute(OpKernelContext* context) override
    {
        // Grabs the input mask.
        const Tensor& mask = context->input(0);
        const Tensor& bcount_dynamic = context->input(1);
        int bNumDims = bcount_dynamic.dims();
        int dim0 = bcount_dynamic.dim_size(0);
        OP_REQUIRES(context, bNumDims == 1 && dim0 == 2,
            errors::InvalidArgument("dynamic_bcount should be one-dimensional with shape[0] == 2."));

        // Grabs input shape.
        int N = mask.dim_size(0);
        int H = mask.dim_size(1);
        int W = mask.dim_size(2);

        bCntH_ = bcount_dynamic.flat<int32>().data()[0];
        bCntW_ = bcount_dynamic.flat<int32>().data()[1];

        // Initializes output.
        // TODO: try to find a way not to redo the allocation in Compute
        Tensor* activeBlockIndices = NULL;
        TensorShape activeBlockShape;
        int maxIndices = N * bCntH_ * bCntW_;
        int activeBlockShapeArr[] = { maxIndices };
        TensorShapeUtils::MakeShape(activeBlockShapeArr, 1, &activeBlockShape);
        OP_REQUIRES_OK(context, context->allocate_output(0, activeBlockShape, &activeBlockIndices));

        unsigned int numBins = 1;
        unsigned int binSize = (maxIndices + numBins - 1) / numBins;
        TensorShape binCountsShape;
        int binCountsShapeArr[] = { (int)numBins };
        TensorShapeUtils::MakeShape(binCountsShapeArr, 1, &binCountsShape);
        Tensor binCounts;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, binCountsShape, &binCounts));

        // Does the computation.
        ReduceMaskFunctor<Device, T>()(
            context->eigen_device<Device>(),          // Device.
            mask.flat<T>().data(),                    // Mask array.
            N,                                        // Batch dimension of the mask.
            H,                                        // Height of the mask.
            W,                                        // Width of the mask.
            tol_,                                     // Threshold for being active.
            bOffsH0_,                                 // Block padding offset height.
            bOffsW0_,                                 // Block padding offset width.
            bSzH_,                                    // Block size height.
            bSzW_,                                    // Block size width.
            bStrH_,                                   // Block stride, height.
            bStrW_,                                   // Block stride, width.
            bCntH_,                                   // Number of blocks, height.
            bCntW_,                                   // Number of blocks, width.
            numBins,
            binSize,
            activeBlockIndices->flat<int64>().data(), // Indices of active blocks.
            binCounts.flat<int32>().data(),           // Indices of active blocks.
            avgpool_
            );

        Tensor* resultOut = nullptr;
        AllocatorAttributes hostAttr; hostAttr.set_on_host(true);
        OP_REQUIRES_OK(context, context->allocate_output(1, binCountsShape, &resultOut, hostAttr));

        // read the resulting block count back from GPU to CPU mem
        if (std::is_same<Device, GPUDevice>::value) {
            cudaMemcpy(&readBack_, binCounts.flat<int32>().data(), sizeof(int32), cudaMemcpyDeviceToHost);
            if (readBack_ == 0) {
                cudaMemset(activeBlockIndices->flat<int64>().data(), 0, sizeof(int64));
                readBack_ = 1;
            }
        } else {
            readBack_ = binCounts.flat<int32>().data()[0];
            if (readBack_ == 0) {
                activeBlockIndices->flat<int64>().data()[0] = 0;
                readBack_ = 1;
            }
        }
        resultOut->flat<int32>().data()[0] = readBack_;
    }

private:
    float tol_ = 0;                  // Active block threshold.
    int bOffsH0_ = 0;                // Block padding offset height, negative.
    int bOffsW0_ = 0;                // Block padding offset width, negative.
    int bSzH_ = 0;                   // Block size height.
    int bSzW_ = 0;                   // Block size width.
    int bStrH_ = 0;                  // Block stride, height.
    int bStrW_ = 0;                  // Block stride, width.
    int bCntH_ = 0;                  // block count in H, zero-padded
    int bCntW_ = 0;                  // block count in W, zero-padded
    bool avgpool_ = 0;
    int readBack_ = 0;
};

// Register the CPU kernels.
#define REGISTER_CPU(T) \
     REGISTER_KERNEL_BUILDER(Name("ReduceMask").Device(DEVICE_CPU).TypeConstraint<T>("T"), ReduceMaskOp<CPUDevice, T>);
REGISTER_CPU(float);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
    REGISTER_KERNEL_BUILDER( \
        Name("ReduceMask") \
        .Device(DEVICE_GPU) \
        .HostMemory("dynamic_bcount") \
        .HostMemory("bin_counts") \
        .TypeConstraint<T>("T"), \
        ReduceMaskOp<GPUDevice, T>);
REGISTER_GPU(float);

#endif // GOOGLE_CUDA

