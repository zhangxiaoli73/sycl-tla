# Steps to build and run 'allgather_gemm_main' example on B60:

### prerequisites
- Intel oneAPI Base Toolkit 2025.3 installed
- ishmem built and installed with Intel MPI backend, e.g., 'cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DENABLE_MPI=ON -DCMAKE_INSTALL_PREFIX=<your ishmem install path> -DCTEST_LAUNCHER=mpi -DBUILD_EXAMPLES=ON -DBUILD_UNIT_TESTS=TRUE
'
- Relative new version of Intel Graphics Compute Runtime, e.g., 'https://github.com/intel/compute-runtime/releases/tag/26.01.36711.4'


### build
- envs for building sycl-tla on B60
```
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export CMAKE_BUILD_TYPE=Release
export IGC_VISAOptions="-perfmodel"
export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export CUTLASS_SYCL_PROFILING_ENABLED=ON # OFF by default
export CC=icx
export CXX=icpx
```
- envs for building sycl-tla with ishmem for this distributed-gemm examples
```
export ISHMEM_PATH=<your ishmem install path>
```
- build sycl-tla along with ishmem
```
cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_SYCL_PROFILING_ENABLED=ON
```
- build distributed-gemm examples
```
ninja test_examples_allgather_gemm_main
```

### run
```
I_MPI_FABRICS=ofi FI_PROVIDER=shm ISHMEM_RUNTIME=MPI mpiexec -np 1 -env ZE_AFFINITY_MASK 0 ./distributed-gemm/build/examples/distributed-gemm/allgather_gemm_main 2048 2048 2048 : -np 1 -env ZE_AFFINITY_MASK 1 ./distributed-gemm/build/examples/distributed-gemm/allgather_gemm_main 2048 2048 2048

```