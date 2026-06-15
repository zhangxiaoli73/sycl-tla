#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
export IGC_VectorAliasBBThreshold=100000000000
export CUTLASS_AR_NUM_REDUCER_WGS=32

# Verify correctness
echo "=== Verification ==="
ZE_AFFINITY_MASK=4,5,6,7 mpirun -np 4 --prepend-rank examples/00_bmg_gemm/00_bmg_gemm_allreduce --m=4096 --n=4096 --k=4096 --verify=1 --debug_log=0

# Benchmark
echo ""
echo "=== Benchmark ==="
ZE_AFFINITY_MASK=4,5,6,7 mpirun -np 4 --prepend-rank examples/00_bmg_gemm/00_bmg_gemm_allreduce --m=4096 --n=4096 --k=4096 --verify=0 --debug_log=0
