
## PyTorch Distributed on Intel XPU: From Native Communication to GEMM-Collective Overlap

Distributed computing is essential for modern AI at scale. We keep distributed support on Intel XPUs in lockstep with PyTorch's distributed evolution by implementing the XCCL backend in TorchComms, delivering oneCCL-powered collectives and symmetric-memory-based custom collectives.

Beyond basic collectives, communication-computation overlap is also a key to reduce communication overhead. We enable asynchronous tensor parallelism on Intel XPUs for transformer-based models, with pipelined AllGather+GEMM and GEMM+ReduceScatter ops across multiple precisions (e.g., FP16, BF16, FP8) for both training and inference.

Moving beyond Python-level fusion ops, we propose a native overlap path built on SYCL-TLA for Intel GPUs. Our approach selects pull or push to match each platform's bandwidth profile, handles pure PCIe topologies alongside high-bandwidth interconnects, and tunes pipeline depth based on workload and platform configuration. When overlap would hurt rather than help, a built-in fallback policy avoids regression. This native design also opens the door to future scale-out support. In practice, we have validated this approach with vLLM serving workloads including Llama and Qwen on Intel Arc B-Series, demonstrating meaningful performance gains especially during the prefill phase.

