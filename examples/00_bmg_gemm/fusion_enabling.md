
# AllGather-GEMM Fusion in PyTorch Inductor

## Call Chain

### Entry: `post_grad.py` — `post_grad_passes()`

```python
# torch/_inductor/fx_passes/post_grad.py
if config._micro_pipeline_tp:
    micro_pipeline_tp_pass(gm.graph)
```

`post_grad_passes()` is the main graph-optimization entry in Inductor. After standard pattern-matching, noop elimination, b2b_gemm, and related passes, it checks `config._micro_pipeline_tp`; if `True`, it invokes `micro_pipeline_tp_pass`.

---

### Core Pass: `micro_pipeline_tp_pass(graph)` — `micro_pipeline_tp.py`

```python
def micro_pipeline_tp_pass(graph: torch.fx.Graph):
    all_gathers = find_all_gather_patterns(graph)
    reduce_scatters = find_reduce_scatter_patterns(graph)

        # If reorder_for_compute_comm_overlap is enabled,
        # prefer simple compute/communication overlap and exclude collectives
        # that can already be hidden.
    if config.reorder_for_compute_comm_overlap:
        unexposed_collectives = _get_unexposed_collectives(graph)
        all_gathers = [x for x in all_gathers if x.ag_node not in unexposed_collectives]
        reduce_scatters = [x for x in reduce_scatters
                           if x.reduce_scatter_node not in unexposed_collectives]

    for all_gather in all_gathers:
        fuse_all_gather_matmul(all_gather)       # ← AllGather-GEMM fusion

    for reduce_scatter in reduce_scatters:
        fuse_matmul_reduce_scatter(reduce_scatter)  # ← GEMM-ReduceScatter fusion
```

---

### AllGather Pattern Matching: `find_all_gather_patterns(graph)`

This function identifies the following four AllGather patterns in the graph (all corresponding to `funcol.all_gather_tensor`):

| Pattern | gather_dim | Signature |
|------|-----------|------|
| `zero_dim_all_gather_pattern` | 0 | `wait_tensor(all_gather_into_tensor(shard, ..., group_name))` |
| `non_zero_dim_all_gather_pattern` | > 0 | `cat([getitem(split(wait_tensor(all_gather(...)), ...), ...), ...])` |
| `zero_dim_type_erased_all_gather_pattern` | 0 | Same as above, but data is transferred as uint8 and restored with `view.dtype` |
| `non_zero_dim_type_erased_all_gather_pattern` | > 0 | Same as above, with additional `view.dtype` wrapping |

The match result is stored as `_AllGatherMatch`, including `shard_node`, `ag_node`, `res_node`, `gather_dim`, and `group_name`.

---

### Fusion Execution: `fuse_all_gather_matmul(all_gather)`

After per-match checks, the pass transforms the following graph pattern:

```
A = all_gather_tensor(A_shard, gather_dim, group_name)   # AllGather
C_0 = torch.matmul(A, B_0)                              # GEMM 0
C_1 = torch.matmul(A, B_1)                              # GEMM 1
...
```

into a single fused operator call:

```python
# Regular matmul
A, [C_0, C_1, ...] = torch.ops.symm_mem.fused_all_gather_matmul(
    A_shard, [B_0, B_1, ...], gather_dim, group_name,
        return_A=True   # Set to False if A has no downstream users
)

# FP8 scaled matmul
A, [C_0, ...] = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
    A_shard, [B_0, ...], A_scale, [B_scale_0, ...],
    gather_dim, group_name, [bias_0, ...], [result_scale_0, ...],
    [out_dtype_0, ...], [use_fast_accum_0, ...]
)
```

After fusion, ancestor nodes of the B matrices are topologically raised so they execute before the fused node, improving compute-communication overlap.

---

## AllGather-GEMM Fusion Conditions

This fusion is triggered only when **all** of the following conditions are satisfied:

### 1. Config Switch
- `config._micro_pipeline_tp = True`
        (This switch controls the entire pass in `post_grad_passes()`.)

### 2. Distributed Environment
- `torch.distributed.is_available()` is True
- `torch.distributed.is_nccl_available()` is True

### 3. Symmetric Memory Is Enabled for the Group
- `is_symm_mem_enabled_for_group(group_name)` is True
        (From `torch.distributed._symmetric_memory`; requires symmetric memory allocation for the communication group.)

### 4. A Matchable AllGather Pattern Exists in the Graph
- The graph contains `all_gather_into_tensor` + `wait_tensor`
        (Supports gather_dim == 0, gather_dim > 0, and uint8 type-erasure variants.)

### 5. AllGather Output Is Consumed by GEMM (as A Matrix)
- The AllGather output `res_node` is directly used as the **left operand (A)** of one or more `aten.mm` / `aten._scaled_mm`
- ND-matmul pattern is supported: `reshape -> mm -> reshape`

### 6. All Consumer GEMMs Have the Same Type
- Either all are regular `_Matmul` (`aten.mm`) or all are `_ScaledMatmul` (`aten._scaled_mm`)
- Mixed types are not allowed

### 7. GEMM Non-A Arguments (B, scales, etc.) Do Not Depend on AllGather Output
- `all_gather.res_node not in matmul.arg_ancestor_nodes`
        (Avoids cyclic dependencies.)

### 8. Additional Constraints When gather_dim Is the Last Dimension
- `shard.shape[-1] >= 1024` (skip when too small; low performance gain)
        - Check location: [`micro_pipeline_tp.py` -> `fuse_all_gather_matmul`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/micro_pipeline_tp.py)
  ```python
  if _is_last_dim(_get_tensor(shard_node), gather_dim):
      if _get_tensor(shard_node).shape[-1] < 1024:
          return
  ```
- `_ScaledMatmul` is not supported (FP8 scaled mm does not support last-dim gather)
        - Same location; filtered by `filter_matmul = _filter_out_scaled_matmul`
- The AllGather result must be used **only by GEMMs**, with no extra users (otherwise explicit materialization is required, adding overhead)
        - Check location: later in the same function
  ```python
  if _is_last_dim(_get_tensor(shard_node), gather_dim) and len(
      all_gather.res_node.users
  ) > len(matmuls):
      return
  ```

### 9. Mutual Exclusion with Compute-Comm Overlap
- If `config.reorder_for_compute_comm_overlap = True` and this AllGather can already be hidden by an `aten.mm` via simple scheduling (unexposed), micro-pipeline TP fusion is **not** applied (prefer the lighter overlap strategy)

---

## Condition Check Flow

```
config._micro_pipeline_tp?
        │ Yes
        ▼
torch.distributed available + xccl available?
        │ Yes
        ▼
is_symm_mem_enabled_for_group(group_name)?
        │ Yes
        ▼
Graph has AllGather pattern (wait_tensor(all_gather_into_tensor(...)))?
        │ Yes
        ▼
AllGather result used as A in aten.mm / aten._scaled_mm?
        │ Yes
        ▼
All consuming GEMMs are same type (Matmul or ScaledMatmul)?
        │ Yes
        ▼
B-args have no dependency on AllGather result?
        │ Yes
        ▼
[If gather_dim == last_dim] shard[-1] >= 1024 AND result only used in GEMMs?
        │ Yes
        ▼
[If reorder_for_compute_comm_overlap] AllGather NOT already unexposed?
        │ Yes
        ▼
→ FUSE: fused_all_gather_matmul / fused_all_gather_scaled_matmul
```

---

## References

- [`torch/_inductor/fx_passes/post_grad.py`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/post_grad.py) - pass entry (`config._micro_pipeline_tp` check, around L213)
- [`torch/_inductor/fx_passes/micro_pipeline_tp.py`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/micro_pipeline_tp.py) - full implementation of AllGather/ReduceScatter fusion
- [`torch/distributed/_symmetric_memory/__init__.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_symmetric_memory/__init__.py) - `is_symm_mem_enabled_for_group` and related striding helpers


