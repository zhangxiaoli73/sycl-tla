
## PyTorch Distributed on Intel XPU: From Native Communication to GEMM-Collective Overlap

Distributed computing is essential for modern AI at scale. We bring distributed support to Intel XPUs by implementing XCCL backend in TorchComms, delivering oneCCL collectives and symmetric-memory custom collectives.
Beyond basic collectives, computation-communication overlap futher reduces overhead. We enable async tensor parallelism on Intel XPUs with pipelined AllGather+GEMM and GEMM+ReduceScatter ops across multiple precisions for training and inference.
Moving beyond Python-level ops, we introduce a native SYCL-TLA overlap path for Intel GPUs. The design selects pull or push by platform bandwidth profile, supports PCIe and high-bandwidth interconnect topologies, tunes pipeline depth by workload, and falls back when overlap is not beneficial. This design also opens the door to future scale-out support. In pratice, we validate this path in vLLM serving (Llama, Qwen) on Intel Arc B-Series, with notable performance gains in initial tests.

