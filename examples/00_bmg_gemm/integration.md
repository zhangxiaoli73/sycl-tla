
1. eager (non inductor backend)
    cutomized op (constructed by multipy kernels - gemm + other communication kernels)
    tuning gemm for tileshape, tileMMA -> host tuning integrated into torch-xpu-ops
    tuning gemm (code link) : runtime tune (warmup too slow) vs. compile tune (kernel binary too large)? SDPA tuning (code link)?

2. inductor
   2 aten ops -> a cutomized op (mulitle kernels - gemm kernel + communition kernel)
    - gemm tuning? sycl-tla backend in torch.compile(...), inductor config? triton tuning?


   cutomized op (constructed by multipy kernels - gemm(stream0) + other communication kernels(stream1))
   tuning gemm for tileshape, tileMMA -> lowering
   sycl tla backend auto tuning (example) -> register decompose ...

   gemm + reduction(host bound) 
