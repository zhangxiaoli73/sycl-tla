/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*
 * Standalone BMG One-Shot Allreduce benchmark.
 *
 * Implementation mirrors bench_oneshot_vs_ccl.cpp for consistent performance:
 *   - Direct exchange_ipc_ptrs (no SymmMemory wrapper overhead)
 *   - zeContextMakeMemoryResident for peer buffers
 *   - q.wait() per iteration (no MPI_Barrier in timed loop)
 *   - Same fused per-WG signal barrier as bench_oneshot_vs_ccl
 *
 * Build (via CMake) or manually:
 *   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 \
 *        -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
 *        -I../examples/00_bmg_gemm -I../include -I../tools/util/include \
 *        -I$I_MPI_ROOT/include -L$I_MPI_ROOT/lib -lmpi -lze_loader \
 *        00_bmg_allreduce_standalone.cpp -o 00_bmg_allreduce_standalone
 *
 * Run:
 *   ZE_AFFINITY_MASK=4,5,6,7 mpirun -np 4 ./00_bmg_allreduce_standalone --m=256 --n=2048
 */

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <level_zero/ze_api.h>

#include "Signal.hpp"
#include "symm.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

// ---------- Constants matching bench_oneshot_vs_ccl.cpp ----------
constexpr int kOneShotMaxNumGroups_AR = 24;
constexpr int kOneShotMaxNumThreads_AR = 256;
constexpr int kFusedSignalBaseU32_AR = 2456;
constexpr int kSignalPadU32Slots_AR = 8192;  // enough for 4 regions × 24 groups × 16 ws
constexpr int kVecBytes = 16;
constexpr int kVecBf16 = kVecBytes / (int)sizeof(bf16);  // 8

template <int N>
struct alignas(kVecBytes) VecBf16 {
  bf16 data[N];
};

// ---------- Fused allreduce kernel (matches bench_oneshot_vs_ccl FusedOneShotKernel) ----------
template <int kWorldSize>
struct FusedOneShotKernel_AR {
  bf16** peer_ptrs;
  bf16* output_ptr;
  uint32_t** signal_pads;
  int64_t numel;
  int my_rank;

  static inline uint32_t* slot_of(
      uint32_t** pads, int owner, int region, int group_id, int src_rank) {
    const int64_t region_off =
        (int64_t)region * kOneShotMaxNumGroups_AR * kWorldSize;
    return pads[owner] + kFusedSignalBaseU32_AR + region_off +
        (int64_t)group_id * kWorldSize + src_rank;
  }

  inline void wg_barrier(sycl::nd_item<1> it, int region) const {
    const auto lid = it.get_local_id(0);
    const auto gid = it.get_group(0);
    if (lid < (size_t)kWorldSize) {
      int peer = (int)lid;
      if (peer != my_rank) {
        uint32_t* put = slot_of(signal_pads, peer, region, gid, my_rank);
        uint32_t* wait = slot_of(signal_pads, my_rank, region, gid, peer);
        put_signal<std::memory_order_release>(put);
        wait_signal<std::memory_order_acquire>(wait);
      }
    }
    it.barrier(sycl::access::fence_space::local_space);
  }

  void operator()(sycl::nd_item<1> it) const {
    wg_barrier(it, /*region=*/0);

    const int64_t tid = (int64_t)it.get_global_linear_id();
    const int64_t stride = (int64_t)it.get_global_range(0);
    const int64_t vec_total = numel / kVecBf16;
    using V = VecBf16<kVecBf16>;

    for (int64_t v = tid; v < vec_total; v += stride) {
      const int64_t e = v * kVecBf16;
      V acc = *reinterpret_cast<const V*>(peer_ptrs[my_rank] + e);
#pragma unroll
      for (int step = 1; step < kWorldSize; ++step) {
        const int p = (my_rank + step) % kWorldSize;
        V rhs = *reinterpret_cast<const V*>(peer_ptrs[p] + e);
#pragma unroll
        for (int i = 0; i < kVecBf16; ++i) {
          acc.data[i] = (bf16)((float)acc.data[i] + (float)rhs.data[i]);
        }
      }
      *reinterpret_cast<V*>(output_ptr + e) = acc;
    }

    it.barrier(sycl::access::fence_space::local_space);
    wg_barrier(it, /*region=*/1);
  }
};

// ---------- Optimized: Reduce-Scatter + Allgather kernel ----------
// Phase 1: Each rank reduces only its 1/world_size chunk (less PCIe traffic)
// Phase 2: Each rank reads the other chunks from the ranks that computed them
// Total cross-device reads: 2 * (world_size-1)/world_size * numel
//   vs all-to-all: (world_size-1) * numel
// For ws=4: 1.5MB vs 3MB cross-device reads
constexpr int kRSAG_MaxGroups = 24;

template <int kWorldSize>
struct ReduceScatterAllgatherKernel {
  bf16** peer_ptrs;      // input data on each rank (read-only in both phases)
  bf16** peer_out_ptrs;  // IPC-visible output buffer per rank (write Phase1, read Phase2)
  bf16* output_ptr;      // local final output
  uint32_t** signal_pads;
  int64_t numel;
  int my_rank;

  static inline uint32_t* slot_of(
      uint32_t** pads, int owner, int region, int group_id, int src_rank) {
    const int64_t region_off =
        (int64_t)region * kRSAG_MaxGroups * kWorldSize;
    return pads[owner] + kFusedSignalBaseU32_AR + region_off +
        (int64_t)group_id * kWorldSize + src_rank;
  }

  inline void wg_barrier(sycl::nd_item<1> it, int region) const {
    const auto lid = it.get_local_id(0);
    const auto gid = it.get_group(0);
    if (lid < (size_t)kWorldSize) {
      int peer = (int)lid;
      if (peer != my_rank) {
        uint32_t* put = slot_of(signal_pads, peer, region, gid, my_rank);
        uint32_t* wait = slot_of(signal_pads, my_rank, region, gid, peer);
        put_signal<std::memory_order_release>(put);
        wait_signal<std::memory_order_acquire>(wait);
      }
    }
    it.barrier(sycl::access::fence_space::local_space);
  }

  // Point-to-point signal: notify one specific peer that we're done
  inline void signal_peer(sycl::nd_item<1> it, int peer, int region) const {
    const auto lid = it.get_local_id(0);
    const auto gid = it.get_group(0);
    if (lid == 0) {
      uint32_t* put = slot_of(signal_pads, peer, region, gid, my_rank);
      put_signal<std::memory_order_release>(put);
    }
  }

  // Point-to-point wait: wait for one specific peer
  inline void wait_for_peer(sycl::nd_item<1> it, int peer, int region) const {
    const auto lid = it.get_local_id(0);
    const auto gid = it.get_group(0);
    if (lid == 0) {
      uint32_t* wait = slot_of(signal_pads, my_rank, region, gid, peer);
      wait_signal<std::memory_order_acquire>(wait);
    }
    it.barrier(sycl::access::fence_space::local_space);
  }

  void operator()(sycl::nd_item<1> it) const {
    // No pre-barrier needed: input data is static across iterations,
    // q.wait() between iterations ensures prior kernel completion.

    const int64_t tid = (int64_t)it.get_global_linear_id();
    const int64_t stride = (int64_t)it.get_global_range(0);
    using V = VecBf16<kVecBf16>;

    // Phase 1: Reduce-scatter
    const int64_t chunk_elems = numel / kWorldSize;
    const int64_t chunk_vecs = chunk_elems / kVecBf16;
    const int64_t my_chunk_start = (int64_t)my_rank * chunk_elems;

    for (int64_t v = tid; v < chunk_vecs; v += stride) {
      const int64_t e = my_chunk_start + v * kVecBf16;
      V acc = *reinterpret_cast<const V*>(peer_ptrs[my_rank] + e);
#pragma unroll
      for (int step = 1; step < kWorldSize; ++step) {
        const int p = (my_rank + step) % kWorldSize;
        V rhs = *reinterpret_cast<const V*>(peer_ptrs[p] + e);
#pragma unroll
        for (int i = 0; i < kVecBf16; ++i) {
          acc.data[i] = (bf16)((float)acc.data[i] + (float)rhs.data[i]);
        }
      }
      // Write reduced chunk to IPC output (serves as both peer-visible AND final output)
      *reinterpret_cast<V*>(output_ptr + e) = acc;
    }

    // Mid-barrier: wait for all peers to finish reduce-scatter
    it.barrier(sycl::access::fence_space::local_space);
    wg_barrier(it, /*region=*/1);

    // Phase 2: Allgather - copy other peers' chunks into our output
    const int64_t total_ag_vecs = chunk_vecs * (kWorldSize - 1);
    for (int64_t flat = tid; flat < total_ag_vecs; flat += stride) {
      const int step = (int)(flat / chunk_vecs) + 1;
      const int64_t v = flat % chunk_vecs;
      const int src = (my_rank + step) % kWorldSize;
      const int64_t e = (int64_t)src * chunk_elems + v * kVecBf16;
      V val = *reinterpret_cast<const V*>(peer_out_ptrs[src] + e);
      *reinterpret_cast<V*>(output_ptr + e) = val;
    }
  }
};

// ---------- Launch config (matches bench_oneshot_vs_ccl init_launch_cfg) ----------
static void get_launch_cfg(int64_t numel, int64_t& groups, int64_t& threads) {
  int64_t total_vec = numel / kVecBf16;
  if (total_vec <= kOneShotMaxNumThreads_AR) {
    groups = 1;
    threads = std::max<int64_t>(32, (total_vec + 31) / 32 * 32);
  } else {
    groups = std::min<int64_t>(
        (total_vec + kOneShotMaxNumThreads_AR - 1) / kOneShotMaxNumThreads_AR,
        (int64_t)kOneShotMaxNumGroups_AR);
    threads = kOneShotMaxNumThreads_AR;
  }
}

static void get_rsag_launch_cfg(int64_t numel, int world_size, int64_t& groups, int64_t& threads) {
  int64_t chunk_vecs = (numel / world_size) / kVecBf16;
  threads = kOneShotMaxNumThreads_AR;
  groups = std::min<int64_t>(
      (chunk_vecs + threads - 1) / threads,
      (int64_t)kRSAG_MaxGroups);
  groups = std::max<int64_t>(groups, 1);
}

// ---------- Options ----------
struct Options {
  bool help = false;
  int m = 8192;
  int n = 4096;
  int iterations = 100;
  int warmup = 20;
  int verify = 1;

  void parse(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      // Support both "--m=256" and "--m 256" formats
      auto get_val = [&](const std::string& key) -> int {
        if (a.rfind(key + "=", 0) == 0) {
          return std::atoi(a.c_str() + key.size() + 1);
        } else if (a == key && i + 1 < argc) {
          return std::atoi(argv[++i]);
        }
        return INT_MIN;
      };
      if (a == "--help" || a == "-h") { help = true; return; }
      int v;
      if ((v = get_val("--m")) != INT_MIN) m = v;
      else if ((v = get_val("--n")) != INT_MIN) n = v;
      else if ((v = get_val("--iterations")) != INT_MIN) iterations = v;
      else if ((v = get_val("--warmup")) != INT_MIN) warmup = v;
      else if ((v = get_val("--verify")) != INT_MIN) verify = v;
    }
  }

  void print_usage() const {
    std::printf(
        "Standalone BMG One-Shot Allreduce (bench-style)\n\n"
        "  --m=<int>            rows (default 8192)\n"
        "  --n=<int>            cols (default 4096)\n"
        "  --iterations=<int>   benchmark iterations (default 100)\n"
        "  --warmup=<int>       warmup iterations (default 20)\n"
        "  --verify=<int>       0/1 verify output (default 1)\n\n");
  }
};

// ---------- main ----------
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  Options options;
  options.parse(argc, argv);
  if (options.help) {
    if (rank == 0) options.print_usage();
    MPI_Finalize();
    return 0;
  }

  int m = options.m;
  int n = options.n;
  int64_t numel = (int64_t)m * n;

  if (numel % kVecBf16 != 0) {
    if (rank == 0)
      std::fprintf(stderr, "Error: m*n must be divisible by %d\n", kVecBf16);
    MPI_Finalize();
    return 1;
  }
  if (numel % (kVecBf16 * world_size) != 0) {
    if (rank == 0)
      std::fprintf(stderr, "Error: m*n must be divisible by %d (vec_width * world_size)\n",
                   kVecBf16 * world_size);
    MPI_Finalize();
    return 1;
  }

  // Select GPU device (one per rank)
  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if ((int)devices.size() < world_size) {
    if (rank == 0)
      std::fprintf(stderr, "Need %d GPUs, found %zu\n", world_size, (size_t)devices.size());
    MPI_Finalize();
    return 1;
  }

  sycl::device dev = devices[rank];
  sycl::context ctx(dev);
  sycl::queue q(ctx, dev, sycl::property_list{sycl::property::queue::in_order{}});

  // Allocate input, output, and signal pad
  bf16* in_buf = sycl::malloc_device<bf16>(numel, q);
  bf16* out_buf = sycl::malloc_device<bf16>(numel, q);
  bf16* ipc_out_buf = sycl::malloc_device<bf16>(numel, q);  // IPC-visible for RS+AG
  uint32_t* sig_buf = sycl::malloc_device<uint32_t>(kSignalPadU32Slots_AR, q);
  q.memset(sig_buf, 0, kSignalPadU32Slots_AR * sizeof(uint32_t)).wait();
  q.memset(ipc_out_buf, 0, numel * sizeof(bf16)).wait();

  // Fill input with rank+1
  q.submit([&](sycl::handler& h) {
    bf16 val = (bf16)(float)(rank + 1);
    bf16* p = in_buf;
    h.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) { p[i] = val; });
  }).wait();
  MPI_Barrier(MPI_COMM_WORLD);

  // IPC exchange (same as bench_oneshot_vs_ccl)
  std::vector<void*> opened_in, opened_sig, opened_out;
  auto peer_in = exchange_ipc_ptrs(in_buf, rank, world_size, q, opened_in);
  auto peer_sig = exchange_ipc_ptrs(sig_buf, rank, world_size, q, opened_sig);
  auto peer_out = exchange_ipc_ptrs(ipc_out_buf, rank, world_size, q, opened_out);

  // Make peer memory resident on local device
  auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
  for (int r = 0; r < world_size; ++r) {
    if (r != rank) {
      zeContextMakeMemoryResident(ze_ctx, ze_dev, peer_in[r], numel * sizeof(bf16));
      zeContextMakeMemoryResident(ze_ctx, ze_dev, peer_out[r], numel * sizeof(bf16));
      zeContextMakeMemoryResident(ze_ctx, ze_dev, peer_sig[r],
                                  kSignalPadU32Slots_AR * sizeof(uint32_t));
    }
  }

  // Upload peer pointer tables to device
  bf16** d_peer_in = sycl::malloc_device<bf16*>(world_size, q);
  bf16** d_peer_out = sycl::malloc_device<bf16*>(world_size, q);
  uint32_t** d_peer_sig = sycl::malloc_device<uint32_t*>(world_size, q);
  {
    std::vector<bf16*> h_in(world_size);
    std::vector<bf16*> h_out(world_size);
    std::vector<uint32_t*> h_sig(world_size);
    for (int r = 0; r < world_size; ++r) {
      h_in[r] = reinterpret_cast<bf16*>(peer_in[r]);
      h_out[r] = reinterpret_cast<bf16*>(peer_out[r]);
      h_sig[r] = reinterpret_cast<uint32_t*>(peer_sig[r]);
    }
    q.memcpy(d_peer_in, h_in.data(), world_size * sizeof(bf16*));
    q.memcpy(d_peer_out, h_out.data(), world_size * sizeof(bf16*));
    q.memcpy(d_peer_sig, h_sig.data(), world_size * sizeof(uint32_t*));
    q.wait();
  }

  // Compute launch config
  int64_t groups, threads;
  get_launch_cfg(numel, groups, threads);

  int64_t rsag_groups, rsag_threads;
  get_rsag_launch_cfg(numel, world_size, rsag_groups, rsag_threads);

  if (rank == 0) {
    std::printf("[allreduce-standalone] m=%d, n=%d, numel=%lld (%lld bytes)\n",
                m, n, (long long)numel, (long long)(numel * sizeof(bf16)));
    std::printf("[allreduce-standalone] all-to-all: groups=%lld, threads=%lld\n",
                (long long)groups, (long long)threads);
    std::printf("[allreduce-standalone] RS+AG:      groups=%lld, threads=%lld (vec=%dB)\n\n",
                (long long)rsag_groups, (long long)rsag_threads, kVecBytes);
  }

  // Lambda to launch kernel (world_size dispatch)
  auto run_alltoall = [&]() {
    auto do_submit = [&](auto ws_const) {
      constexpr int kWS = decltype(ws_const)::value;
      FusedOneShotKernel_AR<kWS> ker{d_peer_in, out_buf, d_peer_sig, numel, rank};
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(groups * threads, threads), ker);
      });
    };
    switch (world_size) {
      case 1:  do_submit(std::integral_constant<int, 1>{}); break;
      case 2:  do_submit(std::integral_constant<int, 2>{}); break;
      case 3:  do_submit(std::integral_constant<int, 3>{}); break;
      case 4:  do_submit(std::integral_constant<int, 4>{}); break;
      case 5:  do_submit(std::integral_constant<int, 5>{}); break;
      case 6:  do_submit(std::integral_constant<int, 6>{}); break;
      case 7:  do_submit(std::integral_constant<int, 7>{}); break;
      case 8:  do_submit(std::integral_constant<int, 8>{}); break;
      default:
        throw std::runtime_error("world_size must be in [1, 8].");
    }
  };

  // Lambda for reduce-scatter + allgather kernel
  auto run_rs_ag = [&]() {
    auto do_submit = [&](auto ws_const) {
      constexpr int kWS = decltype(ws_const)::value;
      ReduceScatterAllgatherKernel<kWS> ker{
          d_peer_in, d_peer_out, ipc_out_buf, d_peer_sig, numel, rank};
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(rsag_groups * rsag_threads, rsag_threads), ker);
      });
    };
    switch (world_size) {
      case 1:  do_submit(std::integral_constant<int, 1>{}); break;
      case 2:  do_submit(std::integral_constant<int, 2>{}); break;
      case 3:  do_submit(std::integral_constant<int, 3>{}); break;
      case 4:  do_submit(std::integral_constant<int, 4>{}); break;
      case 5:  do_submit(std::integral_constant<int, 5>{}); break;
      case 6:  do_submit(std::integral_constant<int, 6>{}); break;
      case 7:  do_submit(std::integral_constant<int, 7>{}); break;
      case 8:  do_submit(std::integral_constant<int, 8>{}); break;
      default:
        throw std::runtime_error("world_size must be in [1, 8].");
    }
  };

  using clk = std::chrono::high_resolution_clock;

  // === Benchmark 1: All-to-all (original) ===
  for (int i = 0; i < options.warmup; ++i) { run_alltoall(); q.wait(); }
  MPI_Barrier(MPI_COMM_WORLD);
  auto t0 = clk::now();
  for (int iter = 0; iter < options.iterations; ++iter) {
    run_alltoall();
    q.wait();
  }
  auto t1 = clk::now();
  double us_alltoall = std::chrono::duration<double, std::micro>(t1 - t0).count() / options.iterations;

  // === Benchmark 2: Reduce-scatter + Allgather ===
  // Reset signal pads between different kernel types
  q.memset(sig_buf, 0, kSignalPadU32Slots_AR * sizeof(uint32_t)).wait();
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < options.warmup; ++i) { run_rs_ag(); q.wait(); }
  MPI_Barrier(MPI_COMM_WORLD);
  t0 = clk::now();
  for (int iter = 0; iter < options.iterations; ++iter) {
    run_rs_ag();
    q.wait();
  }
  t1 = clk::now();
  double us_rs_ag = std::chrono::duration<double, std::micro>(t1 - t0).count() / options.iterations;

  if (rank == 0) {
    double bytes = (double)(numel * sizeof(bf16));
    std::printf("\n=== Results (m=%d, n=%d, %lld bytes, ws=%d) ===\n",
                m, n, (long long)(numel * sizeof(bf16)), world_size);
    std::printf("  All-to-all:              %8.2f us  (%.3f GB/s)\n",
                us_alltoall, bytes / (us_alltoall * 1e-6) / (1ULL << 30));
    std::printf("  Reduce-scatter+Allgather: %8.2f us  (%.3f GB/s)\n",
                us_rs_ag, bytes / (us_rs_ag * 1e-6) / (1ULL << 30));
    std::printf("  Speedup: %.2fx\n\n", us_alltoall / us_rs_ag);
  }

  // Verification
  bool passed = true;
  if (options.verify != 0) {
    std::vector<float> host_gpu(numel);
    {
      std::vector<bf16> tmp(numel);
      q.memcpy(tmp.data(), ipc_out_buf, numel * sizeof(bf16)).wait();
      for (int64_t i = 0; i < numel; ++i)
        host_gpu[i] = (float)tmp[i];
    }

    std::vector<float> host_local(numel, (float)(rank + 1));
    std::vector<float> host_ref(numel);
    MPI_Allreduce(host_local.data(), host_ref.data(),
                  (int)numel, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    size_t mismatch = 0;
    double max_abs = 0.0, max_rel = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
      double ref = (double)host_ref[i];
      double got = (double)host_gpu[i];
      double diff = std::abs(ref - got);
      double rel = (std::abs(ref) > 1e-6) ? diff / std::abs(ref) : diff;
      max_abs = std::max(max_abs, diff);
      max_rel = std::max(max_rel, rel);
      if (rel > 1e-2) ++mismatch;
    }

    passed = (mismatch == 0);
    std::printf("[rank %d] verify %s ref[0]=%.3f gpu[0]=%.3f "
                "max_abs=%.6e max_rel=%.6e mismatch=%zu/%lld\n",
                rank, passed ? "PASSED" : "FAILED",
                host_ref[0], host_gpu[0],
                max_abs, max_rel, mismatch, (long long)numel);
  }

  // Cleanup
  close_ipc_ptrs(q, opened_in);
  close_ipc_ptrs(q, opened_out);
  close_ipc_ptrs(q, opened_sig);
  sycl::free(d_peer_in, q);
  sycl::free(d_peer_out, q);
  sycl::free(d_peer_sig, q);
  sycl::free(in_buf, q);
  sycl::free(out_buf, q);
  sycl::free(ipc_out_buf, q);
  sycl::free(sig_buf, q);

  MPI_Finalize();
  return passed ? 0 : 1;
}
