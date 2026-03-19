export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export CMAKE_BUILD_TYPE=Release
export IGC_VISAOptions="-perfmodel"
export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export CC=icx
export CXX=icpx
 
#export CUTLASS_SYCL_PROFILING_ENABLED=ON # OFF by default
 
#export CMAKE_PREFIX_PATH=/home/sdp/jiafuzha/sycl-tla/apps/ishmem:$CMAKE_PREFIX_PATH
 
export ISHMEM_PATH=/home/sdp/jiafuzha/sycl-tla/apps/ishmem

cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21

ninja test_examples_allgather_gemm_main


