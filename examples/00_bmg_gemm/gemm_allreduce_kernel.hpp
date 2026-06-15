/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <algorithm>
#include <cstring>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "Signal.hpp"

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

// Base offset in signal buffer for reduce-scatter/allgather per-WG barrier.
// Must not overlap with tile completion flags which use offsets [0, world_size * num_tiles).
// Flag layout in signal buffer:
// [0, world_size * num_tiles):          rs_flag (Flag A) — "GEMM tile done, symm_input ready"
// [kAGFlagBaseU32, + world_size * num_tiles): ag_flag (Flag B) — "RS tile done, symm_rs ready"
constexpr int kAGFlagBaseU32 = 4096;
constexpr int kMaxTilesForFlags = 1024;  // supports up to 1024 tiles (32x32 grid)

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

  // IPC pointers to peers' D (output) buffers for allgather phase
  SyclBF16** remote_out_ptrs_dev;  // [world_size] pointers to peer D buffers

  // Mode selection
  bool use_push_mode;         // true=push, false=pull

  // Monotonic per-launch token for flag synchronization.
  // Producer writes signal_token; consumer waits for exact match.
  uint32_t signal_token;

  // Reducer workgroup configuration
  int max_active_reducer_wgs;  // Maximum number of active reducer WGs (for tuning)
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
// Fused GEMM + Allreduce: GEMM WG computes tile, writes to symm_input, pushes rs_flag
/////////////////////////////////////////////////////////////////////////////////////////////////

inline uint32_t allreduce_load_acquire_u32(uint32_t* addr);

template <class ATensor, class BTensor, class CTensor, class TiledMMA>
void gemm_and_push(
    ATensor const& A,       // (M,K) — global tensor
    BTensor const& B,       // (N,K) — global tensor
    CTensor& C_local,       // (M,N) — local data_slots[my_rank], used for copy_c setup
    TiledMMA const& mma,
    float alpha,
    GemmAllreduceParams const& params,
    int tile_m,             // explicit tile row coordinate
    int tile_n) {           // explicit tile col coordinate

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto local_id = int(item.get_local_id(0));

  int wg_m = tile_m;
  int wg_n = tile_n;
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

  // Write to local symm_input_buffer (data_slots[my_rank])
  copy(copy_c, tCrC, tCgC);

  // Group barrier: ensures ALL threads' data writes have completed before flag push.
  sycl::group_barrier(item.get_group());
  // Release fence: ensures data write is globally visible before flag write.
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

  // Push rs_flag to ALL remote ranks and own local flags.
  if (local_id == 0) {
    int tile_linear_id = wg_m * params.num_n_tiles + wg_n;
    int flag_idx = params.my_rank * params.num_m_tiles * params.num_n_tiles + tile_linear_id;

    // Write to own local rs_flag
    params.local_flags[flag_idx] = params.signal_token;

    // Push rs_flag to each remote rank
    for (int peer = 0; peer < params.world_size; ++peer) {
      if (peer == params.my_rank) continue;
      params.remote_flags_dev[peer][flag_idx] = params.signal_token;
    }
  }
  // GEMM WG exits here — no inline RS, no blocking
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// wg_rs: ReduceScatter workgroup
//
// Spin-checks rs_flag for tiles in MY chunk. Once all ranks' flags for a tile are ready,
// pulls remote symm_input data, reduces, writes to local D (symm_rs_buffer), then
// pushes ag_flag to all remote peers.
//
// Tile assignment: static striped across wg_rs WGs.
/////////////////////////////////////////////////////////////////////////////////////////////////

inline uint32_t allreduce_load_acquire_u32(uint32_t* addr);

inline void rs_work(
    sycl::nd_item<2> item,
    int rs_wg_id,
    int num_rs_wgs,
    GemmAllreduceParams const& params,
    bool debug_log = false) {

  int local_id = int(item.get_local_id(0));
  int wg_size = int(item.get_local_range(0));

  if (rs_wg_id >= num_rs_wgs) {
    return;
  }

  constexpr int NUM_PER_TH = 4;
  constexpr int BF16_PER_VEC = NUM_PER_TH * 4;  // 16 bf16 per vector
  using LoadVec = sycl::vec<int64_t, NUM_PER_TH>;
  using BF16Vec = sycl::vec<SyclBF16, BF16_PER_VEC>;

  int num_tiles = params.num_m_tiles * params.num_n_tiles;
  int chunk_m_tiles = params.num_m_tiles / params.world_size;
  int my_chunk_start_row = params.my_rank * chunk_m_tiles;
  int tiles_per_chunk = chunk_m_tiles * params.num_n_tiles;

  int64_t total_elems = static_cast<int64_t>(params.M) * params.N;
  int64_t rank_slot_vec_stride = total_elems / BF16_PER_VEC;

  auto* out_vec = reinterpret_cast<LoadVec*>(params.D);

  // Each wg_rs processes a strided subset of chunk tiles
  for (int tile_idx = rs_wg_id; tile_idx < tiles_per_chunk; tile_idx += num_rs_wgs) {
    int tile_m = my_chunk_start_row + tile_idx / params.num_n_tiles;
    int tile_n = tile_idx % params.num_n_tiles;
    int tile_linear_id = tile_m * params.num_n_tiles + tile_n;

    // Spin-check rs_flag: wait for ALL ranks to have computed this tile
    if (local_id == 0) {
      for (int src = 0; src < params.world_size; ++src) {
        int flag_idx = src * num_tiles + tile_linear_id;
        uint32_t* flag_ptr = &params.local_flags[flag_idx];
        while (allreduce_load_acquire_u32(flag_ptr) != params.signal_token) {
          // spin-wait for rank src's GEMM output for this tile
        }
      }
    }
    sycl::group_barrier(item.get_group());
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);

    // Pull remote data and reduce this tile
    int tile_m_base = tile_m * 256;
    int tile_n_base = tile_n * 256;
    constexpr int tile_rows = 256;
    constexpr int tile_cols = 256;
    int64_t tile_elems = static_cast<int64_t>(tile_rows) * tile_cols;
    int64_t vec_elems = tile_elems / BF16_PER_VEC;  // 4096 vectors per tile

    for (int64_t vi = local_id; vi < vec_elems; vi += wg_size) {
      int64_t bf16_idx = vi * BF16_PER_VEC;
      int row = static_cast<int>(bf16_idx / tile_cols);
      int col = static_cast<int>(bf16_idx % tile_cols);
      int64_t global_offset =
          (static_cast<int64_t>(tile_m_base + row) * params.N) + (tile_n_base + col);
      int64_t global_vec_offset = global_offset / BF16_PER_VEC;

      // Start with my own rank's data (local read)
      auto* buf0 = reinterpret_cast<LoadVec const*>(
          params.remote_data_slots_dev[params.my_rank]);
      BF16Vec sum = buf0[static_cast<int64_t>(params.my_rank) * rank_slot_vec_stride
                         + global_vec_offset].template as<BF16Vec>();

      // Accumulate from remote ranks (ring order for anti-affinity)
      #pragma unroll 8
      for (int step = 1; step < params.world_size; ++step) {
        int src = (params.my_rank + step) % params.world_size;
        auto* buf = reinterpret_cast<LoadVec const*>(
            params.remote_data_slots_dev[src]);
        sum += buf[static_cast<int64_t>(src) * rank_slot_vec_stride
                   + global_vec_offset].template as<BF16Vec>();
      }

      out_vec[global_vec_offset] = sum.template as<LoadVec>();
    }

    // Push ag_flag to ALL remote peers: "my D[tile] is ready for AG"
    sycl::group_barrier(item.get_group());
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    if (local_id == 0) {
      int ag_flag_idx = params.my_rank * num_tiles + tile_linear_id;
      // Write to local ag_flag
      params.local_flags[kAGFlagBaseU32 + ag_flag_idx] = params.signal_token;
      // Push to all remote peers
      for (int peer = 0; peer < params.world_size; ++peer) {
        if (peer == params.my_rank) continue;
        params.remote_flags_dev[peer][kAGFlagBaseU32 + ag_flag_idx] = params.signal_token;
      }
    }
  }

  if (debug_log && local_id == 0 && rs_wg_id == 0) {
    printf("[WG_RS] rank=%d, num_rs_wgs=%d, tiles_per_chunk=%d\n",
           params.my_rank, num_rs_wgs, tiles_per_chunk);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// wg_ag: Allgather workgroup
//
// Spin-checks ag_flag for tiles in OTHER ranks' chunks. Once a remote rank's ag_flag
// for a tile is set, pulls that tile from the remote rank's D into local D.
//
// Tile assignment: static striped across wg_ag WGs.
/////////////////////////////////////////////////////////////////////////////////////////////////

inline void ag_work(
    sycl::nd_item<2> item,
    int ag_wg_id,
    int num_ag_wgs,
    GemmAllreduceParams const& params,
    bool debug_log = false) {

  int local_id = int(item.get_local_id(0));
  int wg_size = int(item.get_local_range(0));

  if (ag_wg_id >= num_ag_wgs) {
    return;
  }

  constexpr int NUM_PER_TH = 4;
  constexpr int BF16_PER_VEC = NUM_PER_TH * 4;
  using LoadVec = sycl::vec<int64_t, NUM_PER_TH>;

  int num_tiles = params.num_m_tiles * params.num_n_tiles;
  int chunk_m_tiles = params.num_m_tiles / params.world_size;
  int tiles_per_chunk = chunk_m_tiles * params.num_n_tiles;

  // Total AG tiles: (world_size - 1) remote chunks
  int total_ag_tiles = tiles_per_chunk * (params.world_size - 1);

  // Each wg_ag processes a strided subset of AG tiles
  for (int ag_idx = ag_wg_id; ag_idx < total_ag_tiles; ag_idx += num_ag_wgs) {
    // Determine which remote rank and which tile
    int step = ag_idx / tiles_per_chunk;  // 0-based (0 = first remote rank)
    int tile_in_chunk = ag_idx % tiles_per_chunk;
    int src = (params.my_rank + step + 1) % params.world_size;

    // Compute tile coordinates
    int src_chunk_start_row = src * chunk_m_tiles;
    int tile_m = src_chunk_start_row + tile_in_chunk / params.num_n_tiles;
    int tile_n = tile_in_chunk % params.num_n_tiles;
    int tile_linear_id = tile_m * params.num_n_tiles + tile_n;

    // Wait for ag_flag from src rank for this tile
    if (local_id == 0) {
      int ag_flag_idx = src * num_tiles + tile_linear_id;
      uint32_t* flag_ptr = &params.local_flags[kAGFlagBaseU32 + ag_flag_idx];
      while (allreduce_load_acquire_u32(flag_ptr) != params.signal_token) {
        // spin-wait: src rank's RS for this tile not done yet
      }
    }
    sycl::group_barrier(item.get_group());
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);

    // Copy this tile from src's D to local D
    int tile_m_base = tile_m * 256;
    int tile_n_base = tile_n * 256;
    constexpr int tile_rows = 256;
    constexpr int tile_cols = 256;
    int64_t tile_elems = static_cast<int64_t>(tile_rows) * tile_cols;
    int64_t vec_elems = tile_elems / BF16_PER_VEC;  // 4096 vectors per tile

    auto* peer_out = reinterpret_cast<LoadVec const*>(params.remote_out_ptrs_dev[src]);
    auto* out_vec = reinterpret_cast<LoadVec*>(params.D);

    for (int64_t vi = local_id; vi < vec_elems; vi += wg_size) {
      int64_t bf16_idx = vi * BF16_PER_VEC;
      int row = static_cast<int>(bf16_idx / tile_cols);
      int col = static_cast<int>(bf16_idx % tile_cols);
      int64_t global_offset =
          (static_cast<int64_t>(tile_m_base + row) * params.N) + (tile_n_base + col);
      int64_t global_vec_offset = global_offset / BF16_PER_VEC;

      out_vec[global_vec_offset] = peer_out[global_vec_offset];
    }
  }

  if (debug_log && local_id == 0 && ag_wg_id == 0) {
    printf("[WG_AG] rank=%d, num_ag_wgs=%d, total_ag_tiles=%d\n",
           params.my_rank, num_ag_wgs, total_ag_tiles);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Legacy reducer_work (kept for reference / fallback)
/////////////////////////////////////////////////////////////////////////////////////////////////

inline void reducer_work(
    sycl::nd_item<2> item,
    int reducer_id,
    GemmAllreduceParams const& params) {

  int total_output_tiles = params.num_m_tiles * params.num_n_tiles;
  int local_id = int(item.get_local_id(0));
  int wg_size = int(item.get_local_range(0));

  // Keep only a small set of reducer workgroups active.
  // This reduces spin-wait pressure from many early-arriving WGs.
  int active_reducer_wgs = std::min(params.num_reducer_wgs, params.max_active_reducer_wgs);
  if (reducer_id >= active_reducer_wgs) {
    return;
  }

  constexpr int NUM_PER_TH = 4;  // 4 × int64_t = 32 bytes = 16 bf16
  constexpr int BF16_PER_VEC = NUM_PER_TH * 4;  // 16 bf16 per vector load

  using LoadVec = sycl::vec<int64_t, NUM_PER_TH>;
  using BF16Vec = sycl::vec<SyclBF16, BF16_PER_VEC>;

  auto* out_vec = reinterpret_cast<LoadVec*>(params.D);
  auto* out_scalar = reinterpret_cast<SyclBF16*>(params.D);
  int64_t rank_slot_vec_stride = (static_cast<int64_t>(params.M) * params.N) / BF16_PER_VEC;
  int64_t rank_slot_elem_stride = static_cast<int64_t>(params.M) * params.N;

  int tile_m_size = 256;
  int tile_n_size = 256;

  for (int tile_idx = reducer_id; tile_idx < total_output_tiles;
       tile_idx += active_reducer_wgs) {
    int tile_m = tile_idx / params.num_n_tiles;
    int tile_n = tile_idx % params.num_n_tiles;

    int tile_m_base = tile_m * tile_m_size;
    int tile_n_base = tile_n * tile_n_size;
    if (tile_m_base >= params.M || tile_n_base >= params.N) {
      continue;
    }

    int tile_rows = std::min(tile_m_size, params.M - tile_m_base);
    int tile_cols = std::min(tile_n_size, params.N - tile_n_base);
    int64_t tile_elems = static_cast<int64_t>(tile_rows) * tile_cols;

    // Step 1: Poll flags — wait for all ranks to complete this tile
    if (local_id == 0) {
      for (int src = 0; src < params.world_size; ++src) {
        int flag_idx = src * params.num_m_tiles * params.num_n_tiles
                     + tile_m * params.num_n_tiles + tile_n;
        // Pull mode: flags are on remote buffer (read from peer)
        uint32_t* flag_ptr = &params.remote_flags_dev[src][flag_idx];
        auto flag_val = allreduce_load_acquire_u32(flag_ptr);
        while (flag_val != params.signal_token) {
          // spin-wait
          flag_val = allreduce_load_acquire_u32(flag_ptr);
        }
      }
    }
    sycl::group_barrier(item.get_group());

    // Step 2: Reduce this tile with one-shot style rank rotation.
    // Vector path handles aligned chunks; lane-0 handles scalar tail.
    int64_t vec_elems = tile_elems / BF16_PER_VEC;

    for (int64_t vi = local_id; vi < vec_elems; vi += wg_size) {
      int64_t bf16_idx = vi * BF16_PER_VEC;
      int row = static_cast<int>(bf16_idx / tile_cols);
      int col = static_cast<int>(bf16_idx % tile_cols);
      int64_t global_offset =
          (static_cast<int64_t>(tile_m_base + row) * params.N) + (tile_n_base + col);
      int64_t global_vec_offset = global_offset / BF16_PER_VEC;

      int src0 = params.my_rank;
      auto* buf0 = reinterpret_cast<LoadVec const*>(params.remote_data_slots_dev[src0]);
      BF16Vec sum = buf0[static_cast<int64_t>(src0) * rank_slot_vec_stride + global_vec_offset]
                        .template as<BF16Vec>();

      // Rank rotation matches one-shot allreduce read order.
      #pragma unroll 8
      for (int step = 1; step < params.world_size; ++step) {
        int src = (params.my_rank + step) % params.world_size;
        auto* buf = reinterpret_cast<LoadVec const*>(params.remote_data_slots_dev[src]);
        sum += buf[static_cast<int64_t>(src) * rank_slot_vec_stride + global_vec_offset]
                   .template as<BF16Vec>();
      }

      out_vec[global_vec_offset] = sum.template as<LoadVec>();
    }

    if (local_id == 0) {
      int64_t scalar_begin = vec_elems * BF16_PER_VEC;
      for (int64_t i = scalar_begin; i < tile_elems; ++i) {
        int row = static_cast<int>(i / tile_cols);
        int col = static_cast<int>(i % tile_cols);
        int64_t global_offset =
            (static_cast<int64_t>(tile_m_base + row) * params.N) + (tile_n_base + col);

        int src0 = params.my_rank;
        auto* buf0 = reinterpret_cast<SyclBF16 const*>(params.remote_data_slots_dev[src0]);
        float acc = static_cast<float>(
            buf0[static_cast<int64_t>(src0) * rank_slot_elem_stride + global_offset]);

        for (int step = 1; step < params.world_size; ++step) {
          int src = (params.my_rank + step) % params.world_size;
          auto* buf = reinterpret_cast<SyclBF16 const*>(params.remote_data_slots_dev[src]);
          acc += static_cast<float>(
              buf[static_cast<int64_t>(src) * rank_slot_elem_stride + global_offset]);
        }
        out_scalar[global_offset] = static_cast<SyclBF16>(acc);
      }
    }

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
    void** ipc_out_ptrs,
    uint32_t signal_token,
    bool use_push_mode,
    int rank,
    int world_size,
    int m,
    int n,
    int num_n_tiles,
    int num_reducer_wgs,
    bool debug_log = false) {

  (void)use_push_mode;

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();

  int num_m_tiles = ceil_div(m, int(get<0>(mma.tile_mnk())));
  int num_gemm_wgs = num_m_tiles * num_n_tiles;

  // Grid layout: GEMM WGs first, then RS WGs, then AG WGs.
  int wg_m = int(item.get_group(1));
  int wg_n = int(item.get_group(0));
  int num_groups_dim0 = int(item.get_group_range(0));
  int linear_wg_id = wg_m * num_groups_dim0 + wg_n;
  int local_id = int(item.get_local_id(0));

  // Split num_reducer_wgs into RS and AG WGs (environment configurable)
  // Default: half RS, half AG. Override with CUTLASS_AR_NUM_RS_WGS.
  int num_rs_wgs = num_reducer_wgs / 2;
  int num_ag_wgs = num_reducer_wgs - num_rs_wgs;

  // Build params
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
  params.num_m_tiles = num_m_tiles;
  params.num_n_tiles = num_n_tiles;
  params.num_reducer_wgs = num_reducer_wgs;
  params.local_data_slots = reinterpret_cast<SyclBF16*>(ipc_data_ptrs[rank]);
  params.remote_data_slots_dev = reinterpret_cast<SyclBF16**>(ipc_data_ptrs);
  params.local_flags = reinterpret_cast<uint32_t*>(ipc_signal_ptrs[rank]);
  params.remote_flags_dev = reinterpret_cast<uint32_t**>(ipc_signal_ptrs);
  params.D = reinterpret_cast<SyclBF16*>(D.data().get());
  params.ldd = stride<0>(D);
  params.remote_out_ptrs_dev = reinterpret_cast<SyclBF16**>(ipc_out_ptrs);
  params.use_push_mode = true;
  params.signal_token = signal_token;
  params.max_active_reducer_wgs = num_reducer_wgs;

  if (linear_wg_id < num_gemm_wgs) {
    // ---- GEMM workgroup: tile interleave order ----
    // Map linear WG id to interleaved tile coordinates (round-robin across chunks)
    int chunk_m_tiles = num_m_tiles / world_size;
    int linear_tile_id = linear_wg_id;
    int tile_n = linear_tile_id % num_n_tiles;
    int tile_row_linear = linear_tile_id / num_n_tiles;  // 0..num_m_tiles-1
    // Interleave: distribute rows across chunks for early RS start
    int chunk_id = tile_row_linear % world_size;
    int row_within_chunk = tile_row_linear / world_size;
    int tile_m = chunk_id * chunk_m_tiles + row_within_chunk;

    if (debug_log && local_id == 0) {
      printf("[AR-GEMM] rank=%d, wg=%d, tile=(%d,%d)\n", rank, linear_wg_id, tile_m, tile_n);
    }
    gemm_and_push(A, B, C, mma, alpha, params, tile_m, tile_n);
  } else if (linear_wg_id < num_gemm_wgs + num_rs_wgs + num_ag_wgs) {
    int comm_wg_id = linear_wg_id - num_gemm_wgs;
    if (comm_wg_id < num_rs_wgs) {
      // ---- RS workgroup ----
      if (debug_log && local_id == 0) {
        printf("[AR-RS] rank=%d, rs_wg_id=%d/%d\n", rank, comm_wg_id, num_rs_wgs);
      }
      rs_work(item, comm_wg_id, num_rs_wgs, params, debug_log);
    } else {
      // ---- AG workgroup ----
      int ag_wg_id = comm_wg_id - num_rs_wgs;
      if (debug_log && local_id == 0) {
        printf("[AR-AG] rank=%d, ag_wg_id=%d/%d\n", rank, ag_wg_id, num_ag_wgs);
      }
      ag_work(item, ag_wg_id, num_ag_wgs, params, debug_log);
    }
  }
  // else: inactive WG, return immediately
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
  #pragma unroll
  for (int r = 1; r < world_size; ++r) {
    auto* buf = reinterpret_cast<const sycl::vec<int64_t, NUM_PER_TH>*>(ipc_data_ptrs[r]);
    sum += buf[vec_idx].template as<sycl::vec<SyclBF16, BF16_VEC_SIZE>>();
  }

  auto* out = reinterpret_cast<sycl::vec<int64_t, NUM_PER_TH>*>(d_ptr);
  out[vec_idx] = sum.template as<sycl::vec<int64_t, NUM_PER_TH>>();
}

inline void allreduce_store_release_u32(uint32_t* addr, uint32_t val) {
  *addr = val;
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

inline uint32_t allreduce_load_acquire_u32(uint32_t* addr) {
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
  return *addr;
}

inline bool one_shot_wait_value(
    uint32_t* addr,
    uint32_t expected,
    int* error_flag,
    uint32_t max_spins) {
  uint32_t spins = 0;
  while (allreduce_load_acquire_u32(addr) != expected) {
    if (spins++ >= max_spins) {
      if (error_flag) {
        *error_flag = 1;
      }
      return false;
    }
  }
  return true;
}

constexpr int kFusedSignalBaseU32 = 512;
constexpr int kOneShotMaxNumGroups = 256;

template <typename T>
constexpr int elems_per_vec() { return 32 / static_cast<int>(sizeof(T)); }

template <typename T, int N>
struct VecT { T data[N]; };

template <typename scalar_t, int kWorldSize>
struct FusedOneShotAllReduceSumKernel {
  static constexpr int kN = elems_per_vec<scalar_t>();
  using Vec = VecT<scalar_t, kN>;

  scalar_t** peer_ptrs;
  scalar_t* output_ptr;
  uint32_t** signal_pads;
  int64_t input_offset;
  int64_t numel;
  int my_rank;

  static inline uint32_t* slot_of(
      uint32_t** signal_pads,
      int owner_rank,
      int region,
      int group_id,
      int src_rank) {
    const int64_t region_off =
        (int64_t)region * kOneShotMaxNumGroups * kWorldSize;
    return signal_pads[owner_rank] + kFusedSignalBaseU32 + region_off +
        (int64_t)group_id * kWorldSize + src_rank;
  }

  inline void wg_barrier_pre(sycl::nd_item<1> item) const {
    const auto lid = item.get_local_id(0);
    const auto group_id = item.get_group(0);
    if (lid < (size_t)kWorldSize) {
      int peer = static_cast<int>(lid);
      if (peer != my_rank) {
        uint32_t* put_addr = slot_of(
            signal_pads, peer, /*region=*/0, group_id, my_rank);
        uint32_t* wait_addr = slot_of(
            signal_pads, my_rank, /*region=*/0, group_id, peer);
        put_signal<std::memory_order_release>(put_addr);
        wait_signal<std::memory_order_acquire>(wait_addr);
      }
    }
    // Gate the non-barrier threads in this WG on the signal exchange above.
    // Local-scope fence is sufficient: the put/wait already issue
    // system-scope atomic_fence(release/acquire) internally, so cross-device
    // memory ordering is already guaranteed.
    item.barrier(sycl::access::fence_space::local_space);
  }

  inline void wg_barrier_post(sycl::nd_item<1> item) const {
    // First ensure all threads in this WG have finished their reads from
    // peer buffers before we signal peers that their buffers are free.
    item.barrier(sycl::access::fence_space::local_space);

    const auto lid = item.get_local_id(0);
    const auto group_id = item.get_group(0);
    if (lid < (size_t)kWorldSize) {
      int peer = static_cast<int>(lid);
      if (peer != my_rank) {
        uint32_t* put_addr = slot_of(
            signal_pads, peer, /*region=*/1, group_id, my_rank);
        uint32_t* wait_addr = slot_of(
            signal_pads, my_rank, /*region=*/1, group_id, peer);
        put_signal<std::memory_order_release>(put_addr);
        wait_signal<std::memory_order_acquire>(wait_addr);
      }
    }
    // No trailing item.barrier: nothing in this WG runs after the post
    // barrier; the kernel exits immediately and the XPU stream provides
    // queue-level ordering for the caller.
  }

  void operator()(sycl::nd_item<1> item) const {
    // pre-barrier: all peers have their buffers filled.
    wg_barrier_pre(item);

    const int64_t tid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t stride = static_cast<int64_t>(item.get_global_range(0));
    const int64_t vec_total = numel / kN;

    // Rank rotation: see OneShotAllReduceSumKernel comment.
    for (int64_t v = tid; v < vec_total; v += stride) {
      const int64_t elem_idx = v * kN + input_offset;
      Vec acc = *reinterpret_cast<const Vec*>(peer_ptrs[my_rank] + elem_idx);
#pragma unroll
      for (int step = 1; step < kWorldSize; ++step) {
        const int p = (my_rank + step) % kWorldSize;
        Vec rhs = *reinterpret_cast<const Vec*>(peer_ptrs[p] + elem_idx);
#pragma unroll
        for (int i = 0; i < kN; ++i) {
          acc.data[i] = static_cast<scalar_t>(
              static_cast<float>(acc.data[i]) +
              static_cast<float>(rhs.data[i]));
        }
      }
      *reinterpret_cast<Vec*>(output_ptr + v * kN) = acc;
    }
    if (tid == 0) {
      for (int64_t i = vec_total * kN; i < numel; ++i) {
        float a = static_cast<float>(peer_ptrs[my_rank][i + input_offset]);
#pragma unroll
        for (int step = 1; step < kWorldSize; ++step) {
          const int p = (my_rank + step) % kWorldSize;
          a += static_cast<float>(peer_ptrs[p][i + input_offset]);
        }
        output_ptr[i] = static_cast<scalar_t>(a);
      }
    }

    // post-barrier: prevent peers from overwriting their buffers before we
    // have finished reading them.
    wg_barrier_post(item);
  }
};

inline void one_shot_signal_sync(
    sycl::nd_item<1> item,
    uint32_t** signal_pads,
    int rank,
    int world_size,
    int* error_flag,
    uint32_t max_spins,
  uint32_t signal_token) {
  int64_t block_id = static_cast<int64_t>(item.get_group_linear_id());
  int64_t num_blocks = static_cast<int64_t>(item.get_group_range(0));
  int lid = static_cast<int>(item.get_local_id(0));

  // Stage A: each block leader marks local completion in its own slot.
  if (lid == 0) {
    uint32_t* local_done = signal_pads[rank] + block_id * static_cast<int64_t>(world_size) + rank;
    allreduce_store_release_u32(local_done, signal_token);
  }
  sycl::group_barrier(item.get_group());

  // Stage B: only block 0 performs one cross-rank handshake per launch.
  if (block_id == 0) {
    if (lid == 0) {
      // Wait until all local blocks have marked completion.
      for (int64_t b = 0; b < num_blocks; ++b) {
        uint32_t* local_done = signal_pads[rank] + b * static_cast<int64_t>(world_size) + rank;
        if (!one_shot_wait_value(local_done, signal_token, error_flag, max_spins)) {
          break;
        }
      }

      // Publish completion to peers and wait for their completion.
      uint32_t peer_token = signal_token + 1;
      for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        uint32_t* put_addr = signal_pads[peer] + rank;
        allreduce_store_release_u32(put_addr, peer_token);
      }
      for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) continue;
        uint32_t* wait_addr = signal_pads[rank] + peer;
        if (!one_shot_wait_value(wait_addr, peer_token, error_flag, max_spins)) {
          break;
        }
      }
    }
    sycl::group_barrier(item.get_group());
  }
}

template <int NUM_PER_TH>
void one_shot_allreduce_device(
    sycl::nd_item<1> item,
    void** ipc_data_ptrs,         // device-side slot pointers [world_size]
    uint32_t** ipc_signal_ptrs,   // device-side signal pad pointers [world_size]
    SyclBF16* local_slot_ptr,     // this rank's symmetric-memory slot
    SyclBF16 const* local_input,  // local input to publish before reduce
    SyclBF16* d_ptr,              // output buffer
    int rank,
    int world_size,
    int64_t n_elems,
    int* error_flag,
    uint32_t signal_token,
    uint32_t max_spins = 100000000u) {

  constexpr int BF16_PER_I64 = 4;
  constexpr int BF16_VEC_SIZE = NUM_PER_TH * BF16_PER_I64;
  static_assert(BF16_VEC_SIZE <= 16, "NUM_PER_TH * 4 must be <= 16 for sycl::vec");

  using VecI64 = sycl::vec<int64_t, NUM_PER_TH>;
  using VecBF16 = sycl::vec<SyclBF16, BF16_VEC_SIZE>;

  const int64_t i64_elems = n_elems / BF16_PER_I64;
  const int64_t vec_elems = i64_elems / NUM_PER_TH;
  const int64_t global_id = static_cast<int64_t>(item.get_global_linear_id());

  // Do not early-return before synchronization. Partial returns can deadlock
  // at work-group barriers inside one_shot_signal_sync.
  bool has_work = (global_id < vec_elems);

  auto const* in_vec = reinterpret_cast<VecI64 const*>(local_input);
  auto* out_vec = reinterpret_cast<VecI64*>(d_ptr);

  if (has_work) {
    // Load local rank's data directly from local_input.
    VecBF16 sum = in_vec[global_id].template as<VecBF16>();

    // Accumulate from remote ranks (rank-rotated to balance access).
    #pragma unroll
    for (int step = 1; step < world_size; ++step) {
      int remote_rank = (rank + step) % world_size;
      auto* buf = reinterpret_cast<VecI64 const*>(ipc_data_ptrs[remote_rank]);
      sum += buf[global_id].template as<VecBF16>();
    }

    out_vec[global_id] = sum.template as<VecI64>();
  }

  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
  one_shot_signal_sync(
      item,
      ipc_signal_ptrs,
      rank,
      world_size,
      error_flag,
      max_spins,
      signal_token);
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
