
# build sycl-tla

source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force

export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=1
export IGC_VISAOptions="-perfmodel"
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export IGC_VectorAliasBBThreshold=100000
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

export CC=icx
export CXX=icpx

export NEOReadDebugKeys=1
export EnablePidFdOrSocketsForIpc=1

rm -rf build/*

cd build

cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DCUTLASS_SYCL_PROFILING_ENABLED=OFF -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 -DCUTLASS_ENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release -DCUTLASS_ENABLE_HEADERS_ONLY=OFF -DCMAKE_CXX_FLAGS="-ftemplate-backtrace-limit=0 -fdiagnostics-color=always"

ninja clean


# build example

ninja 00_bmg_allgather_gemm

ninja 00_bmg_gemm_reducescatter


# run example

mpirun -np 4 --prepend-rank examples/00_bmg_gemm/00_bmg_allgather_gemm --m=8192 --n=1536 --k=4096

mpirun -np 4 --prepend-rank examples/00_bmg_gemm/00_bmg_gemm_reducescatter --m=8192 --n=1536 --k=4096

