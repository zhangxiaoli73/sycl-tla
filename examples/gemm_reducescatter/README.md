# Steps to build and run 'gemm_reducescatter_main' example on BMG:

### Prerequisites
- Intel oneAPI Base Toolkit 2025.3 installed (includes Intel MPI)
- Relatively new version of Intel Graphics Compute Runtime, e.g., https://github.com/intel/compute-runtime/releases/tag/26.01.36711.4

### Build
- Set environment variables for building sycl-tla on BMG:
```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export CMAKE_BUILD_TYPE=Release
export IGC_VISAOptions="-perfmodel"
export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export CC=icx
export CXX=icpx
```
- Configure and build:
```bash
cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21
ninja test_examples_gemm_reducescatter_main
```

### Run
```bash
I_MPI_FABRICS=ofi FI_PROVIDER=shm mpiexec -np 1 -env ZE_AFFINITY_MASK 0 ./build/examples/gemm_reducescatter/gemm_reducescatter_main 2048 2048 2048 : -np 1 -env ZE_AFFINITY_MASK 1 ./build/examples/gemm_reducescatter/gemm_reducescatter_main 2048 2048 2048
```