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
#include <mutex>
#include <omp.h>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include <cuda_runtime.h>

#include "op_utils.h"
#include "sparse_gather.h"

using namespace tensorflow;
using std::cout;
using std::endl;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


// AP TODO: quite a bit of duplication here, refactor
REGISTER_OP("SparseGather")
    .Attr("T: {float}")
    .Attr("bsize: list(int)")
    .Attr("bstride: list(int)")
    .Attr("boffset: list(int)")
    .Attr("transpose: bool = false")
    .Input("x: T")
    .Input("bin_counts: int32")
    .Input("active_block_indices: int64")
    .Output("y: T");

// CPU specialization of actual computation.
// This is a naive CPU implementation, just for testing purpose.
template <typename T> struct SparseGatherFunctor<CPUDevice, T> {
    void operator()(const CPUDevice& d,
        const T* x, int N, int H, int W, int C, T* y,
        int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
        int numActive, const int64_t* activeBlockIndices, bool transpose)
    {
        const int R = bSzH, S = bSzW;
        #pragma omp parallel for
        for (int ib = 0; ib < numActive; ib++) {
            int biN, biH, biW;
            from64Bit(activeBlockIndices[ib], biN, biH, biW);
            int h0 = bOffsH0 + biH * bStrH;
            int w0 = bOffsW0 + biW * bStrW;
            for (int intraBh = 0; intraBh < R; ++intraBh) {
            for (int intraBw = 0; intraBw < S; ++intraBw) {
            for (int cc = 0; cc < C; cc++) {
                int hh = h0 + intraBh;
                int ww = w0 + intraBw;
                T readVal = 0.0f;
                if (hh >= 0 && ww >= 0 && hh < H && ww < W)
                    readVal = x[biN*H*W*C + hh*W*C + ww*C + cc];
                if (transpose) // output to gathered blocks in NCHW
                    y[ib*R*S*C + cc*R*S + intraBh*S + intraBw] = readVal;
                else
                    y[ib*R*S*C + intraBh*S*C + intraBw*C + cc] = readVal;
            } } }
        }
    }
};

// CPU specialization of actual computation.
// This is a naive CPU implementation, just for testing purpose.
template <typename T> struct SparseScatterFunctor<CPUDevice, T> {
    void operator()(const CPUDevice& d,
        const T* x, int N, int H, int W, int C, T* y,
        int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
        int numActive, const int64_t* activeBlockIndices, bool add, bool transpose, bool atomic)
    {
        omp_lock_t writeLock;
        omp_init_lock(&writeLock);

        const int R = bSzH, S = bSzW;
        #pragma omp parallel for
        for (int ib = 0; ib < numActive; ib++) {
            int biN, biH, biW;
            from64Bit(activeBlockIndices[ib], biN, biH, biW);
            for (int intraBh = 0; intraBh < R; ++intraBh) {
            for (int intraBw = 0; intraBw < S; ++intraBw) {
            for (int cc = 0; cc < C; cc++) {
                int h0 = bOffsH0 + biH * bStrH;
                int w0 = bOffsW0 + biW * bStrW;
                int hh = h0 + intraBh;
                int ww = w0 + intraBw;
                T readVal;
                if (transpose)
                    readVal = x[ib*R*S*C + cc*R*S + intraBh*S + intraBw];
                else
                    readVal = x[ib*R*S*C + intraBh*S*C + intraBw*C + cc];
                if (hh >= 0 && ww >= 0 && hh < H && ww < W) {
                    if (add) {
                        omp_set_lock(&writeLock);
                        y[biN * H * W * C + hh * W * C + ww * C + cc] += readVal;
                        omp_unset_lock(&writeLock);
                    } else
                        y[biN*H*W*C + hh*W*C + ww*C + cc] = readVal;
                }
            } } }
        }
    }
};


template<typename T> struct CopyTensorFunctor<CPUDevice, T> {
    void operator()(const CPUDevice&, T* dst, const T* src, int count) {
        #pragma omp parallel for
        for (int i = 0; i < count; i ++)
            dst[i] = src[i];
    }
    const cudaStream_t* getStream(const CPUDevice&) { return nullptr; }
};



// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T> class SparseGatherOp : public OpKernel {
public:
    explicit SparseGatherOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
        // TODO: refactor/remove duplication with other ops
        std::vector<int> bsize, bstride, boffset;
        OP_REQUIRES_OK(context, context->GetAttr("bsize", &bsize));
        OP_REQUIRES_OK(context, context->GetAttr("bstride", &bstride));
        OP_REQUIRES_OK(context, context->GetAttr("boffset", &boffset));
        OP_REQUIRES(context,
            bsize.size() == 2 && bstride.size() == 2 && boffset.size() == 2,
            errors::InvalidArgument("All block attributes must have a shape of (2,)."))
        OP_REQUIRES_OK(context, context->GetAttr("transpose", &transpose_));

        bSzH_    = bsize[0];   bSzW_ = bsize[1];
        bStrH_   = bstride[0]; bStrW_ = bstride[1];
        bOffsH0_ = boffset[0]; bOffsW0_ = boffset[1];
    }

    ~SparseGatherOp() override {
    }

    void Compute(OpKernelContext* context) override
    {
        // Grabs the input mask.
        const Tensor& x = context->input(0);
        OP_REQUIRES(context, x.dims() == 4, errors::InvalidArgument("x must be rank 4"));

        // Grabs input shape.
        int N = x.dim_size(0);
        int H = x.dim_size(1);
        int W = x.dim_size(2);
        int C = x.dim_size(3);

        const Tensor& bin_counts_tensor = context->input(1);
        // read the number of active blocks from bin_counts input that is expected to be always in host mem
        int32 bin0Count = bin_counts_tensor.flat<int32>().data()[0];

        // Initializes output.
        // TODO: try to find a way not to redo the allocation in Compute
        Tensor* y = NULL;
        int yShapeArr[] = { bin0Count, bSzH_, bSzW_, C };
        if (transpose_)
        {
            // output is NCHW for tranposed version
            yShapeArr[1] = C;
            yShapeArr[2] = bSzH_;
            yShapeArr[3] = bSzW_;
        }
        TensorShape yShape;
        TensorShapeUtils::MakeShape(yShapeArr, 4, &yShape);
        OP_REQUIRES_OK(context, context->allocate_output(0, yShape, &y));

        // Does the computation.
        const Tensor& activeBlockIndices = context->input(2);
        SparseGatherFunctor<Device, T>()(
            context->eigen_device<Device>(),
            x.flat<T>().data(), N, H, W, C,
            y->flat<T>().data(),
            bOffsH0_, bOffsW0_, bSzH_, bSzW_, bStrH_, bStrW_,
            bin0Count, (const int64_t*)activeBlockIndices.flat<int64>().data(),
            transpose_);
    }

private:
    int32* counterPtr_ = nullptr;    // host memory for counter result
    int32 counterCpu = 0;            // TODO: hacky
    int bOffsH0_ = 0;                // Block padding offset height, negative.
    int bOffsW0_ = 0;                // Block padding offset width, negative.
    int bSzH_ = 0;                   // Block size height.
    int bSzW_ = 0;                   // Block size width.
    int bStrH_ = 0;                  // Block stride, height.
    int bStrW_ = 0;                  // Block stride, width.
    bool transpose_ = false;
};

// Register the CPU kernels.
#define REGISTER_CPU(T) \
     REGISTER_KERNEL_BUILDER(Name("SparseGather").Device(DEVICE_CPU).TypeConstraint<T>("T"), SparseGatherOp<CPUDevice, T>);
REGISTER_CPU(float);
#undef REGISTER_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
    REGISTER_KERNEL_BUILDER( \
        Name("SparseGather").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("bin_counts"), SparseGatherOp<GPUDevice, T>);
REGISTER_GPU(float);
#undef REGISTER_GPU
#endif


REGISTER_OP("SparseScatterVar")
    .Attr("T: {float}")
    .Attr("bsize: list(int)")
    .Attr("bstride: list(int)")
    .Attr("boffset: list(int)")
    .Attr("add: bool")
    .Attr("atomic: bool = false")
    .Attr("transpose: bool = false")
    .Input("x: T") // Dimensions: bin_counts[0]*bsize[0]*bsize[1]*C
    .Input("bin_counts: int32")
    .Input("active_block_indices: int64")
    .Input("ybase: Ref(T)") // ybase values will be overwritten with scatters from x
    .Output("y: Ref(T)"); // Dimensions: NHWC, scatter will write on top of current y content

REGISTER_OP("SparseScatter")
    .Attr("T: {float}")
    .Attr("bsize: list(int)")
    .Attr("bstride: list(int)")
    .Attr("boffset: list(int)")
    .Attr("add: bool")
    .Attr("atomic: bool = false")
    .Attr("transpose: bool = false")
    .Input("x: T") // Dimensions: bin_counts[0]*bsize[0]*bsize[1]*C
    .Input("bin_counts: int32")
    .Input("active_block_indices: int64")
    .Input("ybase: T") // ybase values will be copied to output and overwritten with scatters from x
    .Output("y: T"); // Dimensions: NHWC, scatter will write on top of ybase content

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T, bool UseVar> class SparseScatterOp : public OpKernel {
public:
    explicit SparseScatterOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
        // TODO: refactor/remove duplication with other ops
        std::vector<int> bsize, bstride, boffset;
        OP_REQUIRES_OK(context, context->GetAttr("bsize", &bsize));
        OP_REQUIRES_OK(context, context->GetAttr("bstride", &bstride));
        OP_REQUIRES_OK(context, context->GetAttr("boffset", &boffset));
        OP_REQUIRES(context,
            bsize.size() == 2 && bstride.size() == 2 && boffset.size() == 2,
            errors::InvalidArgument("All block attributes must have a shape of (2,)."))
        bSzH_    = bsize[0];   bSzW_ = bsize[1];
        bStrH_   = bstride[0]; bStrW_ = bstride[1];
        bOffsH0_ = boffset[0]; bOffsW0_ = boffset[1];
        OP_REQUIRES_OK(context, context->GetAttr("add", &doAdd_));
        OP_REQUIRES_OK(context, context->GetAttr("transpose", &transpose_));
        OP_REQUIRES_OK(context, context->GetAttr("atomic", &atomic_));
        if (!atomic_ && doAdd_) {
            OP_REQUIRES(
                context, bstride[0] >= bsize[0] && bstride[1] >= bsize[1],
                errors::InvalidArgument("Only non-overlapping blocks are supported with add=True, atomic=False") );
        }
    }

    ~SparseScatterOp() override {
    }

    void Compute(OpKernelContext* context) override
    {
        // TODO: learn form TF's implemenation of scatter_nd_op.cc
        // Grabs the input mask.
        const Tensor& x = context->input(0);
        const Tensor& bin_counts_tensor = context->input(1);
        const Tensor& activeBlockIndices = context->input(2);
        //const TensorShape& x = x.shape();
        //OP_REQUIRES(context, x.dims() == 4, errors::InvalidArgument("x must be rank 4"));

        // Grabs input shape.
        Tensor ybase = UseVar ? context->mutable_input(3, false) : context->input(3); // base tensor to overwrite
        const TensorShape& ybaseShape = ybase.shape();
        int N = ybaseShape.dim_size(0);
        int H = ybaseShape.dim_size(1);
        int W = ybaseShape.dim_size(2);
        int C = ybaseShape.dim_size(3);

        OP_REQUIRES(context, ybase.dims() == 4, errors::InvalidArgument("ybase must be rank 4"));

        // read the number of active blocks from bin_counts input that is expected to be always in host mem
        int32 bin0Count = bin_counts_tensor.flat<int32>().data()[0];

        // TODO: verify sizes of x match { bin0Count, bSzH_, bSzW_, C };
        // TODO: try to find a way not to redo the allocation in Compute
        T* outData = nullptr;
        if (UseVar) {
            context->forward_ref_input_to_ref_output(3, 0);
            outData = ybase.flat<T>().data();
        } else {
            // Initializes output.
            Tensor* y = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, ybaseShape, &y));
            T* destData = y->flat<T>().data();
            int sz = y->flat<T>().size();
            const T* srcData = ybase.flat<T>().data();
            CopyTensorFunctor<Device, T>()(context->eigen_device<Device>(), destData, srcData, sz);
            outData = y->flat<T>().data();
        }

        // Splat/add x on top of y
        SparseScatterFunctor<Device, T>()(
            context->eigen_device<Device>(),
            x.flat<T>().data(), N, H, W, C,
            outData,
            bOffsH0_, bOffsW0_, bSzH_, bSzW_, bStrH_, bStrW_,
            bin0Count, (const int64_t*)activeBlockIndices.flat<int64>().data(),
            doAdd_, transpose_, atomic_
        );
    }

private:
    int32* counterPtr_ = nullptr;    // host memory for counter result
    int32 counterCpu = 0;            // TODO: hacky
    int bOffsH0_ = 0;                // Block padding offset height, negative.
    int bOffsW0_ = 0;                // Block padding offset width, negative.
    int bSzH_ = 0;                   // Block size height.
    int bSzW_ = 0;                   // Block size width.
    int bStrH_ = 0;                  // Block stride, height.
    int bStrW_ = 0;                  // Block stride, width.
    bool doAdd_ = false;
    bool transpose_ = false;
    bool atomic_ = false;
};

// Register the CPU kernels.
#define REGISTER_CPU(T) \
    REGISTER_KERNEL_BUILDER( \
        Name("SparseScatter").Device(DEVICE_CPU).TypeConstraint<T>("T"), SparseScatterOp<CPUDevice, T, false>); \
    REGISTER_KERNEL_BUILDER( \
        Name("SparseScatterVar").Device(DEVICE_CPU).TypeConstraint<T>("T"), SparseScatterOp<CPUDevice, T, true>);
REGISTER_CPU(float);
#undef REGISTER_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
    REGISTER_KERNEL_BUILDER( \
        Name("SparseScatter").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("bin_counts"), \
        SparseScatterOp<GPUDevice, T, false>); \
    REGISTER_KERNEL_BUILDER( \
        Name("SparseScatterVar").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("bin_counts"), \
        SparseScatterOp<GPUDevice, T, true>);
REGISTER_GPU(float);
#undef REGISTER_GPU
#endif // GOOGLE_CUDA

// CudaOpTimer implementation
struct CuEventPair {
    CuEventPair() { cudaEventCreate(&event0); cudaEventCreate(&event1); }
    ~CuEventPair() { cudaEventDestroy(event0); cudaEventDestroy(event1); }
    cudaEvent_t event0, event1;
};

static std::map<std::string, std::unique_ptr<CuEventPair>> g_eventsMap;
static std::mutex g_eventsMutex;

REGISTER_OP("CudaOpTimer")
    .Attr("timer_name: string")
    .Attr("is_start: bool")
    .Output("dt: float");

template<typename Device>
class CudaOpTimer : public OpKernel
{
public:
    explicit CudaOpTimer(OpKernelConstruction *context) : OpKernel(context)
    {
        g_eventsMutex.lock(); // in case multiple timers are created simultaneously
        std::string timer_name;
        OP_REQUIRES_OK(context, context->GetAttr("timer_name", &timer_name_));
        OP_REQUIRES_OK(context, context->GetAttr("is_start", &is_start_));

        auto find_events = g_eventsMap.find(timer_name_);
        if (find_events == g_eventsMap.end())
        {
            g_eventsMap[timer_name_] = std::unique_ptr<CuEventPair>(new CuEventPair());
            events_ = g_eventsMap.find(timer_name_)->second.get();
        } else {
            events_ = find_events->second.get();
        }
        g_eventsMutex.unlock();
    }

    void Compute(OpKernelContext *context) override
    {
        const cudaStream_t *stream = (cudaStream_t*)CopyTensorFunctor<Device, float>().getStream(context->eigen_device<Device>());
        float time = -1.0f;
        if (stream) {
            if (is_start_)
            {
                cudaEventRecord(events_->event0, *stream);
            } else {
                cudaEventRecord(events_->event1, *stream);
                cudaEventSynchronize(events_->event1);
                cudaEventElapsedTime(&time, events_->event0, events_->event1);
            }
        }

        Tensor* output = nullptr;
        AllocatorAttributes hostAttr; hostAttr.set_on_host(true);
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output, hostAttr));
        output->scalar<float>()() = time;
    }
    bool         is_start_ = true;
    CuEventPair* events_   = nullptr;
    std::string  timer_name_;
};

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("CudaOpTimer").Device(DEVICE_GPU).HostMemory("dt"), CudaOpTimer<GPUDevice>);
#endif
REGISTER_KERNEL_BUILDER(Name("CudaOpTimer").Device(DEVICE_CPU).HostMemory("dt"), CudaOpTimer<CPUDevice>);

