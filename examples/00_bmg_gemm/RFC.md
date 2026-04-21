# RFC: Computation-Communication Overlap for Distributed GEMM

**Authors:**
- @lzhang2

## **Summary**

PyTorch already contains Python-level overlap patterns in symmetric memory APIs (for example, fused all-gather/matmul and matmul/reduce-scatter logic in `torch/distributed/_symmetric_memory/__init__.py`).

This RFC proposes a **native-layer** implementation for distributed GEMM collectives, focusing on:

- AllGather + GEMM
- GEMM + ReduceScatter

The current Python-side implementation is a strong proof of concept, but it has practical limitations for production use: limited flexibility on platforms with asymmetric read/write bandwidth, relatively high host-side launch overhead, insufficient fallback mechanism when overlap is not beneficial, and limited support for backend-specific implementation strategies.

This proposal moves overlap-critical decision and execution paths into native APIs and keeps Python as a compatibility-preserving API/dispatch layer.


## **Motivation**

This RFC focuses on design and interface choices for a native implementation path under the existing Python API surface.

The implementation direction is driven by practical engineering constraints in the current Python path, especially around runtime read/write policy control, host-side launch overhead and backend-specific capability usage.

### Limitations of the current Python-side implementation

The current Python-side implementation is a strong proof of concept, but it has practical limitations for production-grade overlap:

- Limited flexibility across platforms with asymmetric communication bandwidth:
	- Some platforms have different effective bandwidth for read (pull) vs write (push) modes.
	- The current Python-side implementation does not select between read-mode and write-mode communication, which is not optimal on every platform.
- High host overhead from Python-side launch and dispatch:
	- Frequent launch and coordination steps increase host overhead.
	- Python-side control flow and dispatch overhead can become non-trivial in latency-sensitive paths.
- Insufficient fallback mechanism at runtime:
	- Without robust policy fallback, overlap can regress performance on some shapes.
	- In unfavorable cases, overlap may be slower than non-overlap.
- The Python-side implementation limits implementation flexibility and does not extend naturally to scale-out scenarios:
	- In particular, it makes it difficult to adopt backend-specific implementations such as asynchronous GEMM pipelines built with CUTLASS, which can perform better on some shapes, especially small-M cases. It also does not naturally support computation-communication overlap in scale-out deployments where collectives span multiple nodes.

These limitations motivate moving overlap-critical execution into native backend code while keeping Python-level APIs stable.

## **Proposed Implementation**

### Native API Proposal

This RFC proposes introducing and standardizing native APIs that are consumed by existing Python OPs.

Proposed APIs for AllGather + Matmul:

- `torch.ops.symm_mem._should_use_allgather_matmul_fusion(A_shard, Bs, gather_dim, group_name) -> bool`
- `torch.ops.symm_mem._native_allgather_matmul_fusion_supported(A_shard, Bs, gather_dim, group_name) -> bool`
- `torch.ops.symm_mem._fused_all_gather_matmul_native_impl(A_shard, Bs, gather_dim, group_name, *, return_A=True)`

Proposed APIs for Matmul + ReduceScatter:

- `torch.ops.symm_mem._should_use_matmul_reducescatter_fusion(A, B, reduce_op, scatter_dim, group_name) -> bool`
- `torch.ops.symm_mem._native_matmul_reducescatter_fusion_supported(A, B, reduce_op, scatter_dim, group_name) -> bool`
- `torch.ops.symm_mem._fused_matmul_reducescatter_native_impl(A, B, reduce_op, scatter_dim, group_name)`

API principles:

- Existing user-facing Python APIs remain unchanged.
- Native APIs are internal backend contracts used by Python dispatch.
- Fallback/no-fusion decisions are owned by native policy APIs.

### API and Dispatch Strategy (Python vs Native)

This RFC follows a layered model:

- Python layer:
	- keeps user-facing API shape and semantic contract
	- performs argument validation and lightweight routing
	- consumes native policy decisions and routes accordingly
- Native layer:
	- owns overlap-critical execution and runtime policy
	- exposes backend ops used by Python dispatcher
	- enables platform-specific read/write communication policy selection and backend-specific implementation strategies
	- owns fallback policy logic and exports per-op gate APIs

Interface exposure decision:

- Keep existing user-facing Python APIs unchanged.
- Do not introduce new public user-visible overlap APIs in v1.
- Add/extend internal native backend ops and route to them through existing Python entry points.
- Preserve semantic compatibility and use native policy-driven automatic fallback when overlap is not a good fit.

Selection policy:

- Step 1: Query native gate API to determine whether fusion should be used.
- Step 2: If fusion is allowed, query native support API to determine whether native fusion path should be used.

Pseudo dispatch contract (illustrative):

```python
@torch.library.impl(lib, "fused_all_gather_matmul", "CUDA")
@torch.library.impl(lib, "fused_all_gather_matmul", "XPU")
def fused_all_gather_matmul(A_shard, Bs, gather_dim, group_name, return_A=True):
	# Public API remains unchanged.
	args = validate_and_normalize(A_shard, Bs, gather_dim, group_name, return_A)

	# Step 1: native policy owns fusion/no-fusion decision.
	should_fuse = torch.ops.symm_mem._should_use_allgather_matmul_fusion(
		args.A_shard,
		args.Bs,
		args.gather_dim,
		args.group_name,
	)

	if not should_fuse:
		return _fused_all_gather_matmul_fallback(
			args.A_shard,
			args.Bs,
			args.gather_dim,
			args.group_name,
			return_A=args.return_A,
		)

	# Step 2: native policy decides whether native fusion path is supported.
	native_fusion_supported = torch.ops.symm_mem._native_allgather_matmul_fusion_supported(
		args.A_shard,
		args.Bs,
		args.gather_dim,
		args.group_name,
	)

	if native_fusion_supported:
		# Internal implementation path. Not a new public API.
		return torch.ops.symm_mem._fused_all_gather_matmul_native_impl(
			args.A_shard,
			args.Bs,
			args.gather_dim,
			args.group_name,
			return_A=args.return_A,
		)

	# Compatibility path: existing Python overlap implementation.
	return fused_all_gather_matmul_python_overlap(
		args.A_shard,
		args.Bs,
		args.gather_dim,
		args.group_name,
		return_A=args.return_A,
	)
```

For GEMM + ReduceScatter, the corresponding native gate API is:

```python
torch.ops.symm_mem._should_use_matmul_reducescatter_fusion(...)
torch.ops.symm_mem._native_matmul_reducescatter_fusion_supported(...)
torch.ops.symm_mem._fused_matmul_reducescatter_native_impl(...)
```

This preserves compatibility while enabling native-first optimization.