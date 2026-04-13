/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <algorithm>
#include <cstring>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/mma_atom.hpp"

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;
using SyclBF16 = sycl::ext::oneapi::bfloat16;

/////////////////////////////////////////////////////////////////////////////////////////////////
// MMA selection helpers (adapted from gemm_reducescatter.hpp)
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TA, typename TB, typename TC>
auto choose_mma_op_ar() {
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>)
    return XE_DPAS_TT<8, TC, TA, TB>{};
  else if constexpr (is_same_v<TA, cute::bfloat16_t>)
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  else
    return XE_DPAS_TT<8, float, cute::half_t>{};
}

template <class ATensor, class BTensor, class CTensor>
auto choose_tiled_mma_ar(ATensor const& A, BTensor const& B, CTensor const&) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op_ar<TA, TB, TC>();

  constexpr bool byte = (cute::max(sizeof_bits_v<TA>, sizeof_bits_v<TB>) <= 8);
  constexpr bool a_t = is_constant_v<1, decltype(stride<0>(A))>;
  constexpr bool b_n = is_constant_v<1, decltype(stride<0>(B))>;

  constexpr bool use_1x_dpas_per_k =
      a_t || (byte && b_n);
  constexpr bool use_4x8_sg =
      ((sizeof_bits_v<TB> < sizeof_bits_v<TA>)
       && !(is_same_v<TB, cute::float_e5m2_t>)) ||
      (b_n && sizeof_bits_v<TB> < 8);

  using _K = conditional_t<use_1x_dpas_per_k, C<op.K>, C<op.K * 2>>;

  using WGTile = Shape<_256, _256, _K>;
  using SGLayout8x4 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using SGLayout4x8 = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;
  using SGLayout = conditional_t<use_4x8_sg, SGLayout4x8, SGLayout8x4>;

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;
  return MMA{};
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmAllreduceParams: all kernel parameters for the fused GEMM + Allreduce kernel
/////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmAllreduceParams {
  // GEMM parameters (pointers are raw device pointers)
  void const* A;              // [M, K] bfloat16, RowMajor
  void const* B;              // [K, N] bfloat16, RowMajor (stored as (N,K) ColMajor-ish)
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
  // local_data_slots: pointer to local data_slots[world_size][M][N] (bfloat16)
  SyclBF16* local_data_slots;

  // remote_data_slots[peer]: pointer to data_slots on peer
  // (used in push mode to write to remote)
  SyclBF16** remote_data_slots_dev;  // [world_size] pointers

  // Flags
  uint32_t* local_flags;      // [world_size][num_m_tiles][num_n_tiles]
  uint32_t** remote_flags_dev; // [world_size] pointers to remote flags

  // Output
  SyclBF16* D;                // [M, N] final reduced output
  int ldd;                    // leading dimension of D

  // Mode selection
  bool use_push_mode;         // true=push, false=pull

  // Monotonic per-launch token for flag synchronization.
  // Producer writes signal_token; consumer waits for exact match.
  uint32_t signal_token;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Standalone GEMM device function (writes accumulator → local C buffer)
// Used for verification reference path (unchanged from original)
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class ATensor, class BTensor, class CTensor, class TiledMMA>
void gemm_device(
    ATensor const& A,       // (M,K) — global tensor
    BTensor const& B,       // (N,K) — global tensor
    CTensor& C,             // (M,N) — output buffer
    TiledMMA const& mma,
    float alpha) {

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
  Tensor tCgC = thr_mma.partition_C(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;
  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  clear(tCrC);

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    gemm(mma, tCrA, tCrB, tCrC);
    barrier_wait(barrier_scope);
  }

  using ElementC = typename CTensor::element_type;
  ElementC alpha_elem = static_cast<ElementC>(alpha);
  for (int i = 0; i < size(tCrC); ++i) {
    tCrC(i) = tCrC(i) * alpha_elem;
  }
  copy(copy_c, tCrC, tCgC);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Fused GEMM + Push: GEMM WG computes its tile, then pushes to all peers
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class ATensor, class BTensor, class CTensor, class TiledMMA>
void gemm_and_push(
    ATensor const& A,       // (M,K) — global tensor
    BTensor const& B,       // (N,K) — global tensor
    CTensor& C_local,       // (M,N) — local data_slots[my_rank], used for copy_c setup
    TiledMMA const& mma,
    float alpha,
    GemmAllreduceParams const& params) {

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  // ---- Standard GEMM mainloop (unchanged) ----
  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C_local.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C_local);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
  Tensor tCgC = thr_mma.partition_C(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;
  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  clear(tCrC);

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    gemm(mma, tCrA, tCrB, tCrC);
    barrier_wait(barrier_scope);
  }

  // Apply alpha scaling
  using ElementC = typename CTensor::element_type;
  ElementC alpha_elem = static_cast<ElementC>(alpha);
  for (int i = 0; i < size(tCrC); ++i) {
    tCrC(i) = tCrC(i) * alpha_elem;
  }

  // ---- Write to local data_slots[my_rank] via copy_c ----
  copy(copy_c, tCrC, tCgC);

  // ---- Push to remote peers (push mode) or just signal (pull mode) ----
  int tile_m = int(get<0>(wg_tile));
  int tile_n = int(get<1>(wg_tile));
  int64_t tile_offset = static_cast<int64_t>(wg_m) * tile_m * params.N
                       + static_cast<int64_t>(wg_n) * tile_n;
  int64_t slot_offset = static_cast<int64_t>(params.my_rank) * params.M * params.N;
  int wg_size = int(item.get_local_range(0));

  if (params.use_push_mode) {
    // Push: copy tile data to every remote peer's data_slots[my_rank]
    // Copy path assumes M/N are 256-aligned in launcher checks.

    for (int peer = 0; peer < params.world_size; ++peer) {
      if (peer == params.my_rank) continue;
      SyclBF16* remote_slot = params.remote_data_slots_dev[peer] + slot_offset;
      SyclBF16* local_slot = params.local_data_slots + slot_offset;
      // Fast vectorized copy in row-major order (32B = 16 bf16 per transaction).
      constexpr int NUM_PER_TH = 4;
      constexpr int BF16_PER_VEC = NUM_PER_TH * 4;
      using LoadVec = sycl::vec<int64_t, NUM_PER_TH>;

      int vec_cols = tile_n / BF16_PER_VEC;
      int vec_stride_n = params.N / BF16_PER_VEC;
      int64_t tile_vec_offset = tile_offset / BF16_PER_VEC;
      LoadVec* remote_vec = reinterpret_cast<LoadVec*>(remote_slot);
      LoadVec* local_vec = reinterpret_cast<LoadVec*>(local_slot);

      for (int row = 0; row < tile_m; ++row) {
        int64_t row_vec_base = tile_vec_offset + static_cast<int64_t>(row) * vec_stride_n;
        for (int vc = local_id; vc < vec_cols; vc += wg_size) {
          int64_t vec_off = row_vec_base + vc;
          remote_vec[vec_off] = local_vec[vec_off];
        }
      }
    }
  }

  // Memory fence before setting flags
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

  // Set flags to signal tile completion (only sub-group leader / work-item 0)
  if (local_id == 0) {
    int flag_idx = params.my_rank * params.num_m_tiles * params.num_n_tiles
                 + wg_m * params.num_n_tiles + wg_n;
    if (params.use_push_mode) {
      // Push mode: set flag on every peer's flag buffer
      for (int peer = 0; peer < params.world_size; ++peer) {
        params.remote_flags_dev[peer][flag_idx] = params.signal_token;
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
      }
    } else {
      // Pull mode: set flag on local flag buffer only
      params.local_flags[flag_idx] = params.signal_token;
      sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Reducer WG: polls flags, reduces data_slots, writes to D
/////////////////////////////////////////////////////////////////////////////////////////////////

inline void reducer_work(
    sycl::nd_item<2> item,
    int reducer_id,
    GemmAllreduceParams const& params) {

  int total_output_tiles = params.num_m_tiles * params.num_n_tiles;
  int local_id = int(item.get_local_id(0));
  int wg_size = int(item.get_local_range(0));

  constexpr int NUM_PER_TH = 4;  // 4 × int64_t = 32 bytes = 16 bf16
  constexpr int BF16_PER_VEC = NUM_PER_TH * 4;  // 16 bf16 per vector load

  using LoadVec = sycl::vec<int64_t, NUM_PER_TH>;
  using BF16Vec = sycl::vec<SyclBF16, BF16_PER_VEC>;

  int tile_m_size = 256;
  int tile_n_size = 256;

  for (int tile_idx = reducer_id; tile_idx < total_output_tiles;
       tile_idx += params.num_reducer_wgs) {
    int tile_m = tile_idx / params.num_n_tiles;
    int tile_n = tile_idx % params.num_n_tiles;

    // Step 1: Poll flags — wait for all ranks to complete this tile
    if (local_id == 0) {
      for (int src = 0; src < params.world_size; ++src) {
        int flag_idx = src * params.num_m_tiles * params.num_n_tiles
                     + tile_m * params.num_n_tiles + tile_n;
        uint32_t* flag_ptr;
        if (params.use_push_mode) {
          // Push mode: flags are on local buffer (peers wrote here)
          flag_ptr = &params.local_flags[flag_idx];
        } else {
          // Pull mode: flags are on remote buffer (read from peer)
          flag_ptr = &params.remote_flags_dev[src][flag_idx];
        }
        auto flag_val = *flag_ptr;
        while (flag_val != params.signal_token) {
          // spin-wait
          flag_val = *flag_ptr;
        }
      }
    }
    sycl::group_barrier(item.get_group());

    // Step 2: Vectorized reduction across all world_size data slots for this tile
    int64_t tile_elems = static_cast<int64_t>(tile_m_size) * tile_n_size;
    int64_t vec_elems = tile_elems / BF16_PER_VEC;

    for (int64_t vi = local_id; vi < vec_elems; vi += wg_size) {
      // Compute the bf16 element index within the tile
      int64_t bf16_idx = vi * BF16_PER_VEC;
      int row = static_cast<int>(bf16_idx / tile_n_size);
      int col = static_cast<int>(bf16_idx % tile_n_size);
      int64_t global_offset = (static_cast<int64_t>(tile_m) * tile_m_size + row)
                              * params.N
                              + static_cast<int64_t>(tile_n) * tile_n_size + col;

      // Load from rank 0's slot
      int64_t slot0_offset = 0 * static_cast<int64_t>(params.M) * params.N + global_offset;
      SyclBF16* slot0_ptr;
      if (params.use_push_mode) {
        slot0_ptr = params.local_data_slots;
      } else {
        slot0_ptr = params.remote_data_slots_dev[0];
      }
      LoadVec raw0 = reinterpret_cast<LoadVec*>(slot0_ptr + slot0_offset)[0];
      BF16Vec sum = raw0.template as<BF16Vec>();

      // Accumulate from remaining ranks
      for (int src = 1; src < params.world_size; ++src) {
        int64_t slot_offset = static_cast<int64_t>(src) * params.M * params.N + global_offset;
        SyclBF16* slot_ptr;
        if (params.use_push_mode) {
          slot_ptr = params.local_data_slots;
        } else {
          slot_ptr = params.remote_data_slots_dev[src];
        }
        LoadVec raw = reinterpret_cast<LoadVec*>(slot_ptr + slot_offset)[0];
        sum += raw.template as<BF16Vec>();
      }

      // Store reduced result to D
      reinterpret_cast<LoadVec*>(params.D + global_offset)[0] =
          sum.template as<LoadVec>();
    }

    sycl::group_barrier(item.get_group());

    // No flag clear is needed with token-based synchronization.
  }
}

template <class ATensor, class BTensor, class CTensor, class DTensor, class TiledMMA>
void gemm_allreduce_device(
    ATensor const& A,
    BTensor const& B,
    CTensor& C,
    DTensor& D,
    TiledMMA const& mma,
    float alpha,
    int** ipc_signal_ptrs,
    void** ipc_data_ptrs,
    uint32_t signal_token,
    bool use_push_mode,
    int rank,
    int world_size,
    int m,
    int n,
    int num_n_tiles) {

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  int reducer_id = int(item.get_group(1)) * num_n_tiles + int(item.get_group(0));

  GemmAllreduceParams params{};
  params.A = static_cast<void const*>(A.data().get());
  params.B = static_cast<void const*>(B.data().get());
  params.M = m;
  params.N = n;
  params.K = shape<1>(A);
  params.lda = stride<0>(A);
  params.ldb = stride<1>(B);
  params.alpha = alpha;
  params.world_size = world_size;
  params.my_rank = rank;
  params.num_m_tiles = ceil_div(m, int(get<0>(mma.tile_mnk())));
  params.num_n_tiles = num_n_tiles;
  params.num_reducer_wgs = params.num_m_tiles * params.num_n_tiles;
  params.local_data_slots = reinterpret_cast<SyclBF16*>(ipc_data_ptrs[rank]);
  params.remote_data_slots_dev = reinterpret_cast<SyclBF16**>(ipc_data_ptrs);
  params.local_flags = reinterpret_cast<uint32_t*>(ipc_signal_ptrs[rank]);
  params.remote_flags_dev = reinterpret_cast<uint32_t**>(ipc_signal_ptrs);
  params.D = reinterpret_cast<SyclBF16*>(D.data().get());
  params.ldd = stride<0>(D);
  params.use_push_mode = use_push_mode;
  params.signal_token = signal_token;

  gemm_and_push(A, B, C, mma, alpha, params);
  sycl::group_barrier(item.get_group());
  reducer_work(item, reducer_id, params);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Standalone Allreduce kernel (kept for verification / separate-kernel path)
/////////////////////////////////////////////////////////////////////////////////////////////////

template <int NUM_PER_TH>
void allreduce_device(
    sycl::nd_item<1> item,
    void** ipc_data_ptrs,    // device-side data pointers [world_size]
    SyclBF16* d_ptr,         // output buffer
    int world_size,
    int64_t n_elems) {

  constexpr int BF16_PER_I64 = 4;
  constexpr int BF16_VEC_SIZE = NUM_PER_TH * BF16_PER_I64;
  static_assert(BF16_VEC_SIZE <= 16, "NUM_PER_TH * 4 must be <= 16 for sycl::vec");

  const int64_t i64_elems = n_elems / BF16_PER_I64;
  const int64_t vec_elems = i64_elems / NUM_PER_TH;
  const int64_t vec_idx = static_cast<int64_t>(item.get_global_linear_id());
  if (vec_idx >= vec_elems) return;

  // Load first peer buffer
  auto* buf0 = reinterpret_cast<const sycl::vec<int64_t, NUM_PER_TH>*>(ipc_data_ptrs[0]);
  auto sum = buf0[vec_idx].template as<sycl::vec<SyclBF16, BF16_VEC_SIZE>>();

  // Accumulate remaining peers
  for (int r = 1; r < world_size; ++r) {
    auto* buf = reinterpret_cast<const sycl::vec<int64_t, NUM_PER_TH>*>(ipc_data_ptrs[r]);
    sum += buf[vec_idx].template as<sycl::vec<SyclBF16, BF16_VEC_SIZE>>();
  }

  auto* out = reinterpret_cast<sycl::vec<int64_t, NUM_PER_TH>*>(d_ptr);
  out[vec_idx] = sum.template as<sycl::vec<int64_t, NUM_PER_TH>>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel name tags
/////////////////////////////////////////////////////////////////////////////////////////////////
template <class TA, class TB>
class GemmKernelName;

template <class TA, class TB>
class GemmAllreduceKernelName;

template <int NUM_PER_TH>
class AllreduceKernelName;

class FusedGemmAllreduceKernel;
