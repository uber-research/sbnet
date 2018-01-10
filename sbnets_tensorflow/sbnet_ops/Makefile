#
#   Sparse Blocks Networks
#

.PHONY: all clean test

all: build/libsbnet.so

clean:
	rm -f build/*.o
	rm -f build/*.so

test:
	cd ../benchmark && bash run_all_unittests.bash # unit tests
	cd tests && python test_sparse_blocks.py # integration tests

CUDA_INC = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64
TF_INC   = $(shell python -c "import tensorflow; print(tensorflow.sysconfig.get_include())")
GCC5     = $(shell g++ --version | grep 5.4 | wc -l)
ifeq ($(GCC5), 1)
ABI      =-D_GLIBCXX_USE_CXX11_ABI=0
else
ABI      =
endif
#@echo Setting ABI=$(ABI)

NVCCFLAGS=-O3 -use_fast_math -std=c++11 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_52,code=sm_52 \
	-I $(TF_INC) -I $(CUDA_INC) -D GOOGLE_CUDA=1 -x cu -w -Xcompiler -fPIC --expt-relaxed-constexpr \
	$(ABI) # -lineinfo

CUOBJS=\
	build/reduce_mask.cu.o \
	build/sparse_gather.cu.o

build/%.cu.o: %.cu
	mkdir -p build
	nvcc $(NVCCFLAGS) -c $< -o $@

build/libsbnet.so: $(CUOBJS)
	ln -sf build/libsbnet.so
	mkdir -p build
	g++ -std=c++11 -g -shared \
		sparse_gather.cc reduce_mask.cc \
		-o build/libsbnet.so \
		build/sparse_gather.cu.o build/reduce_mask.cu.o \
		-I $(TF_INC) \
		-I $(CUDA_INC) \
		-D GOOGLE_CUDA=1 -fPIC -lcudart -L $(CUDA_LIB) $(ABI)

