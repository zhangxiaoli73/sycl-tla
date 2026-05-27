
## PyTorch Distributed on Intel XPU: From Native Communication to GEMM-Collective Overlap

Intel client GPUs like Arc B-series grow more powerful and support larger models, driving the needs for distributed execution. We enable this on Intel client GPUs (integrated and discrete) via an XCCL backend in TorchComms, providing oneCCL-based and custom symmetric-memory collectives.
We further reduce communication overhead via computation–communication overlap, enabling asynchronous tensor parallelism on Intel GPUs with pipelined AllGather+GEMM and GEMM+ReduceScatter ops for training and inference across multiple precisions.
Moving beyond Python ops, we introduce a native SYCL-TLA based overlap path for Intel GPUs. It uses a bandwidth-aware heuristic to choose pull or push, supports PCIe and high-bandwidth interconnects, adapts pipeline depth by workloads, and falls back when overlap isn’t beneficial, etc. This design can be extended to scale-out support on future Intel Crescent Island. We validate it in vLLM serving (Llama, Qwen) on Intel Arc B-series GPUs, showing notable gains.

## Symmetric-Memory Runtime for Overlapping Communication and Computation on Intel XPU Platforms

Communication overhead is a key scalability bottleneck in distributed LLM workloads. We present a symmetric-memory-based communication runtime on Intel XPU platforms enabling communication–computation overlap via lightweight sync, async pipelines, fused kernels, and IBGDA-based GPU-initiated communication, improving hardware utilization and reducing overhead. Specifically, we implement pipelined AllGather+GEMM and GEMM+ReduceScatter using SYCL-TLA, and fused AllGather+Permute and Unpermute+ReduceScatter for MoE workloads to reduce memory movement and improve transfer efficiency. We also explore an IBGDA-based symmetric-memory runtime for lower-latency GPU-initiated communication. Building on these primitives, our runtime performs bandwidth- and topology-aware scheduling to select fast paths, adapt pipeline depth, and tune workgroup configuration. Experiments with vLLM serving (Llama, Qwen) on Intel Arc B-series GPUs show notable performance improvements.

## PyTorch Distributed on Intel XPU: From Collectives to Communication-Computation Overlap
As model and serving workloads continue to grow, efficient distributed execution on Intel GPUs has become increasingly important.We enable distributed execution on Intel GPUs through an XCCL backend integrated with both native PyTorch c10d and the next-generation TorchComms stack, offering oneCCL-based collectives alongside custom symmetric-memory collectives. We support diverse parallelism strategies across practical deployment scenarios, aligned with community needs and PyTorch evolution.

Beyond collective communication, we reduce end-to-end overhead through communication-computation overlap, enabling asynchronous tensor parallelism, plus kernel fusion for sparse MoE workloads.

We deployed this stack at Argonne National Laboratory on Intel® Data Center GPUs systems, validating native XCCL for large-scale HPC workflows and yielding bring-up practices that improved deployment stability and shortened time to production.

