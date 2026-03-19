

I_MPI_FABRICS=ofi FI_PROVIDER=shm ISHMEM_RUNTIME=MPI mpiexec -np 1 -env ZE_AFFINITY_MASK 7 /home/sdp/jiafuzha/sycl-tla/distributed-gemm/build/examples/distributed-gemm/allgather_gemm_main 2048 1536 4096 : -np 1 -env ZE_AFFINITY_MASK 6  /home/sdp/jiafuzha/sycl-tla/distributed-gemm/build/examples/distributed-gemm/allgather_gemm_main 2048 1536 4096

