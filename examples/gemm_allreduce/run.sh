

I_MPI_FABRICS=ofi FI_PROVIDER=shm mpiexec -np 1 -env ZE_AFFINITY_MASK 7 ./build/examples/gemm_reducescatter/gemm_reducescatter_main 2048 1536 4096 : -np 1 -env ZE_AFFINITY_MASK 6 ./build/examples/gemm_reducescatter/gemm_reducescatter_main 2048 1536 4096

