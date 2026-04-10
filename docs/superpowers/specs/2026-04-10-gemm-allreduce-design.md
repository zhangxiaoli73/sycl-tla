# Fused GEMM + Allreduce Kernel Design for Intel Xe20 (BMG)

**Date:** 2026-04-10
**Status:** Draft
**Target Hardware:** Intel Arc B580 (BMG, Xe20), 160 Xe Cores
**Scenario:** LLM Decode, Tensor Parallelism (TP=4), PCIe P2P interconnect

---

## Motivation

In LLM decode with small batch sizes, GEMM is memory-bound (M=64~256, skinny matrices) and allreduce is latency-bound (total data ~1 MB, PCIe bandwidth is sufficient). Separating GEMM and allreduce into two kernel launches adds fixed latency (kernel launch overhead + barrier synchronization) that dominates the allreduce time. A fused single-kernel design eliminates this overhead by overlapping allreduce communication with GEMM computation at tile granularity.

**Key insight:** Multi-stream pipelining (Option B from earlier discussion) does not help here because each additional chunk adds fixed barrier + launch latency. Only a fused kernel can reduce the total latency.

**Design philosophy:** Space-for-time trade-off — allocate `world_size × M×N` symmetric memory so each GEMM work-group can push its output tile directly to all peers, while dedicated reducer work-groups poll for completed tiles and perform the reduction immediately.

---

## Section 1: Memory Layout & IPC Buffer Structure

### Data Buffers

Each rank allocates a symmetric data buffer of size `world_size × M × N × sizeof(bfloat16)`:

```
data_slots[world_size][M][N]   // bfloat16
```

- `data_slots[r]` is the slot where **rank r** writes its GEMM output.
- In **push mode**: GEMM WG on rank `r` writes its tile to `data_slots[r]` on **every** peer (including local).
- In **pull mode**: GEMM WG on rank `r` writes only to local `data_slots[r]`. Reducers on each peer read remote `data_slots[r]` when the flag is set.
- Both push and pull modes are retained; performance testing determines the default.

### Flag Buffers

Each rank allocates a flag buffer for tile-level signaling:

```
flags[world_size][num_m_tiles][num_n_tiles]   // uint32_t, atomically accessed
```

- `flags[src_rank][tile_m][tile_n]` indicates whether rank `src_rank` has completed writing tile `(tile_m, tile_n)`.
- **Value 0:** tile not ready. **Value 1:** tile data available.
- Flag buffers are IPC-exchanged so that:
  - In push mode: the writing rank sets remote flags on each peer atomically.
  - In pull mode: the writing rank sets its own local flag; reducers on other ranks read remote flags.

### Output Buffer

```
D[M][N]   // bfloat16, final allreduced result
```

Written by reducer WGs after summing all `world_size` slots for each tile.

### Memory Layout Diagram

```
Rank 0:                          Rank 1:
┌────────────────────────┐      ┌────────────────────────┐
│ data_slots[0][M][N]    │◄──── │ GEMM WG writes here    │  (push mode)
│ data_slots[1][M][N]    │      │ data_slots[0][M][N]    │
│ data_slots[2][M][N]    │      │ data_slots[1][M][N]    │◄─── (own output)
│ data_slots[3][M][N]    │      │ data_slots[2][M][N]    │
├────────────────────────┤      │ data_slots[3][M][N]    │
│ flags[4][m_tiles][n_t] │      ├────────────────────────┤
├────────────────────────┤      │ flags[4][m_tiles][n_t] │
│ D[M][N]  (output)      │      ├────────────────────────┤
└────────────────────────┘      │ D[M][N]  (output)      │
                                └────────────────────────┘
```

### Alignment Requirements

- All data buffers must be 128-byte aligned (Intel Xe 2D block copy requirement).
- `M` and `N` must be divisible by the WG tile size (256). Tail tiles are not handled — assert/error if not aligned.
- Flag buffer entries are `uint32_t`, naturally aligned.

---

## Section 2: Kernel Launch Structure

### 2D nd_range Convention

The kernel uses a 2D `nd_range` matching the sycl-tla GEMM convention:

- **dim0 (x):** Work-items within a WG + N tile dispatch. Local size = `size(mma)` (e.g., 512 for 8×4 SG layout × 16 subgroup_size).
- **dim1 (y):** M tile index, extended with extra rows for reducer WGs

```
local  = {size(mma), 1}      // e.g., {512, 1}
global = {size(mma) * num_n_tiles, 1 * (num_m_tiles + num_reducer_rows)}
```

This matches the existing GEMM launch pattern where `local = {size(mma), 1}` and tiles are dispatched via `get_group(0)` (N tiles) and `get_group(1)` (M tiles).

### Work-Group Role Dispatch

Inside the kernel, each WG determines its role based on its M-dimension index:

```cpp
int wg_m = item.get_group(1);  // M-dim WG index
int wg_n = item.get_group(0);  // N-dim WG index

if (wg_m < num_m_tiles) {
    // === GEMM WG ===
    gemm_and_push(wg_m, wg_n, ...);
} else {
    // === Reducer WG ===
    int reducer_id = (wg_m - num_m_tiles) * num_n_tiles + wg_n;
    reducer_work(reducer_id, ...);
}
```

### Reducer WG Count Calculation

Reducer WGs occupy "spare" Xe Cores not used by GEMM:

```cpp
constexpr int BMG_XE_CORES = 160;
constexpr int MAX_REDUCER_WGS = 8;  // tunable cap

int num_gemm_wgs = num_m_tiles * num_n_tiles;
int num_reducer_wgs = std::min(MAX_REDUCER_WGS, BMG_XE_CORES - num_gemm_wgs);

// Reducer WGs must not steal Xe Cores from GEMM
assert(num_reducer_wgs > 0 && "Not enough spare Xe Cores for reducers");
```

- For M=256, N=4096 with 256×256 tiles: `num_gemm_wgs = 1 × 16 = 16`, spare cores = 144, `num_reducer_wgs = min(8, 144) = 8`.
- For M=64, N=4096: `num_gemm_wgs = 1 × 16 = 16`, spare = 144, `num_reducer_wgs = 8`.
- With 160 Xe Cores, spare capacity is abundant for typical skinny-M shapes. The assert guards against pathological cases where `num_gemm_wgs >= BMG_XE_CORES`.

### Extra nd_range Rows for Reducers

```cpp
int num_reducer_rows = (num_reducer_wgs + num_n_tiles - 1) / num_n_tiles;
```

Reducer WGs that map beyond `total_tiles = num_m_tiles * num_n_tiles + num_reducer_wgs` early-exit.

---

## Section 3: GEMM WG Push Logic

### GEMM Mainloop

The GEMM mainloop is unchanged from the existing sycl-tla GEMM:

1. Setup 2D block copies for A, B (via `make_xe_2d_copy` / `make_block_2d_copy`)
2. K-loop with prefetch depth=3, `barrier_arrive / barrier_wait`
3. Accumulate in `tCrC` (fp32 accumulators in registers)
4. Apply `alpha` scaling

### Post-GEMM: Push to Peers

After the GEMM mainloop completes for tile `(wg_m, wg_n)`:

```
┌─────────────────────────────────────────────┐
│ 1. Convert tCrC (fp32) → bfloat16           │
│ 2. Write to local data_slots[my_rank]       │
│    at offset (wg_m * 256 * N + wg_n * 256)  │
│ 3. [Push mode only] For each peer p:        │
│    Write to remote_data_slots[p][my_rank]   │
│    at same tile offset via IPC pointer       │
│ 4. Memory fence (release semantics)          │
│ 5. Set flags:                                │
│    [Push] atomic_store remote_flags[p]       │
│           [my_rank][wg_m][wg_n] = 1          │
│    [Pull] atomic_store local_flags           │
│           [my_rank][wg_m][wg_n] = 1          │
└─────────────────────────────────────────────┘
```

### Atomic Ordering

```cpp
// After writing tile data to all destinations:
sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

// Set flag (push mode — on remote peer's flag buffer):
sycl::atomic_ref<uint32_t,
    sycl::memory_order::release,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>
    flag_ref(remote_flags_ptr[peer][my_rank][tile_m][tile_n]);
flag_ref.store(1);
```

The `memory_scope::system` ensures visibility across PCIe-connected devices.

### Copy to Remote via 2D Block Store

The tile copy to remote IPC buffers reuses the existing `make_block_2d_copy_D` pattern from sycl-tla, operating on the remote pointer instead of local D:

```cpp
// For each peer p (push mode):
auto remote_tile_ptr = remote_data_slots[p][my_rank] + tile_offset;
// Use copy_c pattern: copy tCrC → remote_tile_ptr
```

---

## Section 4: Reducer WG Logic

### Tile Assignment

Each reducer WG is assigned tiles in a round-robin fashion:

```cpp
int total_output_tiles = num_m_tiles * num_n_tiles;
for (int tile_idx = reducer_id; tile_idx < total_output_tiles; tile_idx += num_reducer_wgs) {
    int tile_m = tile_idx / num_n_tiles;
    int tile_n = tile_idx % num_n_tiles;
    reduce_tile(tile_m, tile_n, ...);
}
```

### Per-Tile Reduction

For each assigned tile `(tile_m, tile_n)`:

```
┌──────────────────────────────────────────────┐
│ 1. Poll: for each src_rank in [0..world_size)│
│    Spin on flags[src_rank][tile_m][tile_n]   │
│    using atomic_ref acquire until == 1       │
│ 2. group_barrier(group) — all WG threads     │
│    see consistent flag state                 │
│ 3. Vectorized reduction:                     │
│    Load data_slots[0..world_size-1] for tile │
│    Sum using .as<sycl::vec<BF16, N>>()       │
│ 4. Store result to D at tile offset          │
│ 5. Clear flags:                              │
│    flags[src_rank][tile_m][tile_n] = 0       │
│    (for next iteration reuse)                │
└──────────────────────────────────────────────┘
```

### Flag Polling

```cpp
// Each sub-group leader polls one flag
for (int src = 0; src < world_size; ++src) {
    sycl::atomic_ref<uint32_t,
        sycl::memory_order::acquire,
        sycl::memory_scope::system,
        sycl::access::address_space::global_space>
        flag_ref(flags_ptr[src][tile_m][tile_n]);

    while (flag_ref.load() == 0) {
        // spin-wait
    }
}
sycl::group_barrier(item.get_group());
```

### Vectorized Reduction (`.as<>()` Pattern)

Reuses the proven pattern from `test_local_reduction.cpp`:

```cpp
constexpr int NUM_PER_TH = 4;  // 4 × int64_t = 32 bytes = 16 bf16
constexpr int BF16_PER_VEC = NUM_PER_TH * 4;  // 16 bf16 per thread

using LoadVec = sycl::vec<int64_t, NUM_PER_TH>;
using BF16Vec = sycl::vec<sycl::ext::oneapi::bfloat16, BF16_PER_VEC>;

int tile_elems = 256 * 256;  // elements in one tile
int vec_elems = tile_elems / BF16_PER_VEC;
int wg_size = item.get_local_range(0); // size(mma), e.g. 512

for (int vi = local_id; vi < vec_elems; vi += wg_size) {
    LoadVec raw0 = reinterpret_cast<LoadVec*>(slot0 + tile_offset)[vi];
    BF16Vec sum = raw0.template as<BF16Vec>();

    for (int src = 1; src < world_size; ++src) {
        LoadVec raw = reinterpret_cast<LoadVec*>(slot_ptr[src] + tile_offset)[vi];
        sum += raw.template as<BF16Vec>();
    }

    reinterpret_cast<LoadVec*>(D + tile_offset)[vi] = sum.template as<LoadVec>();
}
```

### Reducer WG Size

Reducer WGs share the same local range `{size(mma), 1}` as GEMM WGs (e.g., 512 work-items). All work-items participate in flat vectorized reduction work — the MMA SG layout is unused by reducers, but the nd_range shape must be consistent across all WGs in the kernel.

---

## Section 5: SymmMemory Extension & Host-side Initialization

### SymmMemory Changes

The existing `SymmMemory` class in `symm.hpp` needs the following extensions:

| Current | New |
|---------|-----|
| Single `M×N` data buffer per rank | `world_size × M × N` data buffer per rank |
| Signal pads for barrier only | Additional `flags` buffer for tile-level signaling |
| IPC exchange for data + signal pads | IPC exchange for data + flags |

#### New Allocation

```cpp
// Data buffer: world_size slots, each M×N bf16
size_t data_bytes = world_size * M * N * sizeof(bfloat16);
local_data_ptr_ = sycl::malloc_device<char>(data_bytes, queue);

// Flags buffer: world_size × num_m_tiles × num_n_tiles × sizeof(uint32_t)
size_t flags_bytes = world_size * num_m_tiles * num_n_tiles * sizeof(uint32_t);
local_flags_ptr_ = sycl::malloc_device<uint32_t>(flags_bytes, queue);

// Zero-initialize flags
queue.memset(local_flags_ptr_, 0, flags_bytes).wait();
```

#### IPC Exchange for Flags

The existing `exchange_ipc_ptrs()` pattern is reused for the flags buffer:

1. Get L0 IPC handle for `local_flags_ptr_`
2. Exchange handles via MPI + Unix domain socket (fd passing)
3. Open remote handles → `remote_flags_ptrs_[peer]`
4. Upload to device buffer → `remote_flags_ptrs_dev_`

### GemmAllreduceParams Struct

All kernel parameters are packed into a single struct passed to the kernel:

```cpp
struct GemmAllreduceParams {
    // GEMM parameters
    ElementA const* A;          // [M, K]
    ElementB const* B;          // [K, N]
    int M, N, K;
    int lda, ldb;               // leading dimensions
    float alpha;

    // Distributed parameters
    int world_size;
    int my_rank;
    int num_m_tiles;            // M / 256
    int num_n_tiles;            // N / 256
    int num_reducer_wgs;

    // IPC data pointers (on device)
    // local_data_slots: pointer to local data_slots[world_size][M][N]
    bfloat16* local_data_slots;

    // remote_data_slots[peer]: pointer to data_slots on peer
    // (used in push mode to write to remote)
    bfloat16** remote_data_slots_dev;  // [world_size] pointers

    // Flags
    uint32_t* local_flags;      // [world_size][num_m_tiles][num_n_tiles]
    uint32_t** remote_flags_dev; // [world_size] pointers to remote flags

    // Output
    bfloat16* D;                // [M, N] final reduced output
    int ldd;                    // leading dimension of D

    // Mode selection
    bool use_push_mode;         // true=push, false=pull
};
```

### Host-Side Launch Code

```cpp
void run_fused_gemm_allreduce(sycl::queue& q, const GemmAllreduceParams& params,
                               int mma_size /* = size(mma), e.g. 512 */) {
    int num_reducer_rows = (params.num_reducer_wgs + params.num_n_tiles - 1)
                           / params.num_n_tiles;

    auto local  = sycl::range<2>(mma_size, 1);
    auto global = sycl::range<2>(
        mma_size * params.num_n_tiles,
        1 * (params.num_m_tiles + num_reducer_rows)
    );

    q.submit([&](sycl::handler& h) {
        h.parallel_for<GemmAllreduceKernel>(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(16)]] {
                int wg_m = item.get_group(1);
                int wg_n = item.get_group(0);

                if (wg_m < params.num_m_tiles) {
                    gemm_and_push(item, wg_m, wg_n, params);
                } else {
                    int reducer_id = (wg_m - params.num_m_tiles)
                                     * params.num_n_tiles + wg_n;
                    if (reducer_id < params.num_reducer_wgs) {
                        reducer_work(item, reducer_id, params);
                    }
                    // else: early exit (padding WG)
                }
            }
        );
    });
}
```

### Pre-Launch Assertions

```cpp
assert(M % 256 == 0 && "M must be divisible by WG tile size 256");
assert(N % 256 == 0 && "N must be divisible by WG tile size 256");
assert(num_reducer_wgs > 0 && "Not enough spare Xe Cores for reducer WGs");
assert(reinterpret_cast<uintptr_t>(A) % 128 == 0 && "A must be 128-byte aligned");
assert(reinterpret_cast<uintptr_t>(B) % 128 == 0 && "B must be 128-byte aligned");
```

---

## Push vs Pull Mode Comparison

| Aspect | Push Mode | Pull Mode |
|--------|-----------|-----------|
| **Data writer** | GEMM WG writes to all peers | GEMM WG writes local only |
| **Data reader** | Reducer reads local | Reducer reads remote |
| **PCIe writes** | `(world_size - 1)` remote writes per GEMM WG | 0 remote writes per GEMM WG |
| **PCIe reads** | 0 remote reads per reducer tile | `(world_size - 1)` remote reads per reducer tile |
| **GEMM latency** | Higher (remote writes in critical path) | Lower (local write only) |
| **Reducer latency** | Lower (data already local) | Higher (remote reads needed) |
| **Expected winner** | Small world_size, fast PCIe | Large world_size, slow PCIe |

Both modes share the same flag signaling mechanism. The `use_push_mode` field in `GemmAllreduceParams` selects at runtime.

---

## Files to Modify

| File | Change |
|------|--------|
| `examples/00_bmg_gemm/gemm_allreduce_kernel.hpp` | Rewrite: `gemm_and_push()`, `reducer_work()`, `GemmAllreduceParams` |
| `examples/00_bmg_gemm/00_bmg_gemm_allreduce.cpp` | Rewrite: fused launch, param setup, verification |
| `examples/00_bmg_gemm/symm.hpp` | Extend: `world_size × M×N` allocation, flags buffer, IPC exchange |

---

## Out of Scope (v1)

- Tail tile handling (M or N not divisible by 256) — assert/error
- Automatic fallback to separate GEMM + allreduce when no spare Xe Cores
- Multi-channel pipelining (single iteration only)
- PVC support (BMG-specific Xe Core count and intrinsics)
- Overlapping multiple GEMM+allreduce invocations

---

## Testing Strategy

1. **Correctness:** Compare fused output D against: single-rank GEMM → MPI_Allreduce reference (existing pattern in `00_bmg_gemm_allreduce.cpp`)
2. **Push vs Pull:** Run both modes, compare latency
3. **Tile sizes:** Test with M=256, N=4096 (sweet spot) and M=64, N=4096 (extreme skinny)
4. **Flag mechanics:** Unit test for flag set/poll cycle between 2 ranks (extend `test_local_reduction.cpp`)
