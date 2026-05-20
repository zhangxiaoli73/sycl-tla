/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cute/tensor.hpp>

#include "cutlass/util/command_line.h"
#include "gemm_allreduce_kernel.hpp"
#include "symm.hpp"

using Element = bfloat16_t;

struct Options {
  bool help = false;
  int m = 8192;
  int n = 4096;
  int iterations = 20;
  int verify = 1;
  int wg_size = 0;      // 0 = auto
  int max_blocks = 0;   // 0 = auto

  void parse(int argc, char** argv) {
    std::vector<char const*> cargs(argc);
    for (int i = 0; i < argc; ++i) {
      cargs[i] = argv[i];
    }

    cutlass::CommandLine cmd(argc, cargs.data());
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 8192);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("verify", verify, 1);
    cmd.get_cmd_line_argument("wg_size", wg_size, 0);
    cmd.get_cmd_line_argument("max_blocks", max_blocks, 0);
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Standalone BMG One-Shot Allreduce Example\\n\\n"
        << "  --m=<int>            rows (default 8192)\\n"
        << "  --n=<int>            cols (default 4096)\\n"
        << "  --iterations=<int>   benchmark iterations (default 20)\\n"
        << "  --verify=<int>       0/1 verify output (default 1)\\n"
        << "  --wg_size=<int>      work-group size, 0=auto (default 0)\\n"
        << "  --max_blocks=<int>   launch blocks cap, 0=auto (default 0)\\n\\n"
        << "Constraints:\\n"
        << "  - This one-shot path is vector-only. m * n must be divisible by alignment/sizeof(bf16).\\n"
        << "    With current build-time alignment=32 bytes, m * n must be divisible by 16.\\n\\n";
    return out;
  }
};

template <int NUM_PER_TH>
class StandaloneOneShotAllreduceKernelName;

// ─── RMS Norm Kernel ───────────────────────────────────────────────────────────
// Applies per-row RMS normalization: out[i,j] = in[i,j] / rms(row_i) * weight[j]
// where rms(row) = sqrt(mean(x^2) + eps).  Each work-group handles one row.
template <typename T>
struct RmsNormKernel {
  T const* input;
  T const* weight;
  T*       output;
  int64_t  m;
  int64_t  n;
  float    eps;

  void operator()(sycl::nd_item<1> item) const {
    int64_t row     = static_cast<int64_t>(item.get_group(0));
    int64_t tid     = static_cast<int64_t>(item.get_local_id(0));
    int64_t wg_size = static_cast<int64_t>(item.get_local_range(0));

    if (row >= m) return;

    T const* row_in  = input  + row * n;
    T*       row_out = output + row * n;

    // Accumulate partial sum of squares.
    float partial_ss = 0.0f;
    for (int64_t j = tid; j < n; j += wg_size) {
      float x = static_cast<float>(row_in[j]);
      partial_ss += x * x;
    }

    // Work-group reduction.
    sycl::group<1> g = item.get_group();
    float total_ss = sycl::reduce_over_group(g, partial_ss, sycl::plus<float>{});
    float rms_inv  = sycl::rsqrt(total_ss / static_cast<float>(n) + eps);

    // Write normalized output.
    for (int64_t j = tid; j < n; j += wg_size) {
      float x = static_cast<float>(row_in[j]);
      float w = static_cast<float>(weight[j]);
      row_out[j] = static_cast<T>(x * rms_inv * w);
    }
  }
};

struct Runner {
  std::unique_ptr<sycl::queue> q_;
  std::unique_ptr<SymmMemory> symm_;

  template <int NUM_PER_TH>
    void launch_one_shot_allreduce(
      Element** ipc_data_ptrs,
      uint32_t** ipc_signal_ptrs,
      Element* local_slot, //local symm data
      Element const* local_input,
      Element* out,
      int* dev_error,
      int rank,
      int world_size,
      int64_t n_elems,
      uint32_t signal_token,
      int cfg_wg_size,
      int cfg_max_blocks) {

    constexpr int BF16_PER_I64 = 4;
    constexpr int BF16_VEC = NUM_PER_TH * BF16_PER_I64;  // bf16 elements per thread vector op

    if ((n_elems % BF16_VEC) != 0) {
      throw std::runtime_error(
        "n_elems must be divisible by NUM_PER_TH * 4 for one-shot vector allreduce.");
    }

    // Thread sizing (inspired by CUDA one-shot: target ~512 threads to limit register pressure)
    int dev_max_wg = static_cast<int>(
        q_->get_device().get_info<sycl::info::device::max_work_group_size>());
    int auto_wg_size = std::min(dev_max_wg, 256);
    int wg_size = (cfg_wg_size > 0) ? std::min(dev_max_wg, cfg_wg_size) : auto_wg_size;
    if (wg_size <= 0) {
      throw std::runtime_error("Invalid wg_size after device clamp.");
    }

    int64_t vec_elems = n_elems / BF16_VEC;
    int64_t blocks = std::max<int64_t>(1, (vec_elems + wg_size - 1) / wg_size);
    int auto_max_blocks = std::max(1, static_cast<int>(
      q_->get_device().get_info<sycl::info::device::max_compute_units>()) * 8);
    int max_blocks = (cfg_max_blocks > 0) ? cfg_max_blocks : auto_max_blocks;
    blocks = std::min<int64_t>(blocks, std::min<int64_t>(max_blocks, kOneShotMaxNumGroups));
    int64_t global_size = blocks * wg_size;
    std::cout << "Launching one-shot allreduce with global_size=" << global_size
              << " wg_size=" << wg_size
              << " blocks=" << blocks
              << std::endl;
    
    auto do_submit = [&](auto ws_const) {
      constexpr int kWS = decltype(ws_const)::value;
      q_->submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(global_size)),
                sycl::range<1>(static_cast<size_t>(wg_size))),
            FusedOneShotAllReduceSumKernel<Element, kWS>{
                ipc_data_ptrs,
                out,
                ipc_signal_ptrs,
                /*input_offset=*/0,
                n_elems,
                rank});
      });
    };

    switch (world_size) {
      case 1:  do_submit(std::integral_constant<int,  1>{}); break;
      case 2:  do_submit(std::integral_constant<int,  2>{}); break;
      case 3:  do_submit(std::integral_constant<int,  3>{}); break;
      case 4:  do_submit(std::integral_constant<int,  4>{}); break;
      case 5:  do_submit(std::integral_constant<int,  5>{}); break;
      case 6:  do_submit(std::integral_constant<int,  6>{}); break;
      case 7:  do_submit(std::integral_constant<int,  7>{}); break;
      case 8:  do_submit(std::integral_constant<int,  8>{}); break;
      case 9:  do_submit(std::integral_constant<int,  9>{}); break;
      case 10: do_submit(std::integral_constant<int, 10>{}); break;
      case 11: do_submit(std::integral_constant<int, 11>{}); break;
      case 12: do_submit(std::integral_constant<int, 12>{}); break;
      case 13: do_submit(std::integral_constant<int, 13>{}); break;
      case 14: do_submit(std::integral_constant<int, 14>{}); break;
      case 15: do_submit(std::integral_constant<int, 15>{}); break;
      case 16: do_submit(std::integral_constant<int, 16>{}); break;
      default:
        throw std::runtime_error(
            "FusedOneShotAllReduceSumKernel: world_size must be in [1, 16].");
    }
  }

  // ─── RMS Norm launcher ─────────────────────────────────────────────────────
  void launch_rms_norm(
      Element const* input,
      Element const* weight,
      Element*       output,
      int64_t        m,
      int64_t        n,
      float          eps,
      int            cfg_wg_size) {

    int dev_max_wg = static_cast<int>(
        q_->get_device().get_info<sycl::info::device::max_work_group_size>());
    // Choose wg_size to cover the row (up to dev max), rounded down to power of 2.
    int raw = (cfg_wg_size > 0) ? std::min(dev_max_wg, cfg_wg_size)
                                : std::min(dev_max_wg, static_cast<int>(n));
    int wg_size = 1;
    while (wg_size * 2 <= raw) wg_size *= 2;

    q_->submit([&](sycl::handler& h) {
      h.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(static_cast<size_t>(m) * static_cast<size_t>(wg_size)),
              sycl::range<1>(static_cast<size_t>(wg_size))),
          RmsNormKernel<Element>{input, weight, output, m, n, eps});
    });
  }

  bool run(Options const& options, sycl::device const& device, int rank, int world_size) {
    int m = options.m;
    int n = options.n;
    int64_t n_elems = static_cast<int64_t>(m) * n;

    if (!q_) {
      auto ctx = sycl::context(device);
      q_ = std::make_unique<sycl::queue>(
          ctx, device,
          sycl::property_list{sycl::property::queue::in_order{},
                              sycl::property::queue::enable_profiling{}});
    }

    size_t per_rank_elems = static_cast<size_t>(m) * n;
    size_t total_data_elems = per_rank_elems * world_size;

    // one_shot_signal_sync indexes signal pad as [block_id * world_size + rank],
    // so signal pad storage must cover all launched blocks.
    constexpr int ALIGNMENT_BYTES = 32;
    constexpr int BF16_PER_I64 = 4;
    constexpr int ELEM_PER_THREAD = ALIGNMENT_BYTES / sizeof(Element);
    static_assert((ELEM_PER_THREAD % 4) == 0,
                  "ELEM_PER_THREAD must be divisible by 4 for VecI64 packing");
    constexpr int NUM_PER_TH = ELEM_PER_THREAD / 4;
    constexpr int BF16_VEC = NUM_PER_TH * BF16_PER_I64;

    if ((n_elems % BF16_VEC) != 0) {
      throw std::runtime_error(
          "n_elems must be divisible by NUM_PER_TH * 4 for one-shot vector allreduce.");
    }

    int dev_max_wg = static_cast<int>(
        q_->get_device().get_info<sycl::info::device::max_work_group_size>());
    int auto_wg_size = std::min(dev_max_wg, 256);
    int wg_size = (options.wg_size > 0) ? std::min(dev_max_wg, options.wg_size) : auto_wg_size;
    if (wg_size <= 0) {
      throw std::runtime_error("Invalid wg_size after device clamp.");
    }
    int auto_max_blocks = std::max(1, static_cast<int>(
        q_->get_device().get_info<sycl::info::device::max_compute_units>()) * 8);
    int max_blocks = (options.max_blocks > 0) ? options.max_blocks : auto_max_blocks;
    int64_t vec_elems = n_elems / BF16_VEC;
    int64_t required_blocks = std::max<int64_t>(1, (vec_elems + wg_size - 1) / wg_size);
    int64_t launch_blocks = std::min<int64_t>(required_blocks, std::min<int64_t>(max_blocks, kOneShotMaxNumGroups));
    size_t signal_pad_elems = static_cast<size_t>(kFusedSignalBaseU32) +
                              2u * static_cast<size_t>(kOneShotMaxNumGroups) * static_cast<size_t>(world_size);
    std::cout << "Symm data elements: " << total_data_elems << ", signal pad elemements: " << signal_pad_elems << std::endl;

    if (!symm_) {
      symm_ = std::make_unique<SymmMemory>(
          m, n, 1, rank, world_size, *q_, 8,
          total_data_elems, signal_pad_elems);
    }

    // one-shot handshake protocol assumes signal pads start from 0 on first launch.
    // Re-initialize local signal pad for this run.
    q_->memset(symm_->local_signal_ptr_, 0, signal_pad_elems * sizeof(uint32_t)).wait();
    MPI_Barrier(MPI_COMM_WORLD);

    Element* local_data = reinterpret_cast<Element*>(symm_->local_data_ptr_);
    Element* local_slot = local_data + static_cast<size_t>(rank) * per_rank_elems;
    Element* local_input = sycl::malloc_device<Element>(static_cast<size_t>(n_elems), *q_);
    Element* out = sycl::malloc_device<Element>(static_cast<size_t>(n_elems), *q_);
    int* dev_error = sycl::malloc_device<int>(1, *q_);

    if (local_input == nullptr || out == nullptr || dev_error == nullptr) {
      throw std::runtime_error("Failed to allocate local_input/output/dev_error buffer.");
    }

    // Keep slots clean for debug visibility; one-shot kernel publishes local_input to local_slot.
    q_->memset(local_data, 0, total_data_elems * sizeof(Element)).wait();
    q_->submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<1>(static_cast<size_t>(n_elems)), [=](sycl::id<1> i) {
        local_input[i] = static_cast<Element>(static_cast<float>(rank + 1));
      });
    });
    q_->wait();
    MPI_Barrier(MPI_COMM_WORLD);

    // Build per-rank slot-adjusted device pointer array.
    // remote_data_ptrs_[r] is the IPC-mapped base of peer r's local_data_ptr_.
    // Rank r wrote its data at base + r * per_rank_elems, so we adjust each pointer
    // so that slot_ptrs[r] points directly at the data that rank r produced.
    std::vector<Element*> host_slot_ptrs(world_size);
    for (int r = 0; r < world_size; ++r) {
      uint8_t* base = reinterpret_cast<uint8_t*>(symm_->remote_data_ptrs_[r]);
      host_slot_ptrs[r] = reinterpret_cast<Element*>(
          base + static_cast<size_t>(r) * per_rank_elems * sizeof(Element));
    }
    Element** dev_slot_ptrs = sycl::malloc_device<Element*>(static_cast<size_t>(world_size), *q_);
    if (dev_slot_ptrs == nullptr) {
      throw std::runtime_error("Failed to allocate dev_slot_ptrs.");
    }
    q_->memcpy(dev_slot_ptrs, host_slot_ptrs.data(),
               static_cast<size_t>(world_size) * sizeof(Element*)).wait();

    static_assert(ALIGNMENT_BYTES % sizeof(Element) == 0,
            "ALIGNMENT_BYTES must be divisible by element size");

    if (rank == 0) {
      std::cout << "[one-shot] alignment=" << ALIGNMENT_BYTES
                << "B, elements-per-thread=" << ELEM_PER_THREAD
                << ", NUM_PER_TH=" << NUM_PER_TH
                << ", wg_size=" << (options.wg_size > 0 ? options.wg_size : std::min(static_cast<int>(q_->get_device().get_info<sycl::info::device::max_work_group_size>()), 256))
                << ", max_blocks=" << (options.max_blocks > 0 ? options.max_blocks : std::max(1, static_cast<int>(q_->get_device().get_info<sycl::info::device::max_compute_units>()) * 8))
                << std::endl;
    }

    uint32_t signal_token = 1;
    std::cout << "[rank " << rank << "] Starting one-shot allreduce warmup..." << std::endl;
    for (int i = 0; i < 5; ++i) {
      q_->memset(dev_error, 0, sizeof(int)).wait();
      launch_one_shot_allreduce<NUM_PER_TH>(
          dev_slot_ptrs,
          symm_->remote_signal_ptrs_dev_,
          local_slot,
          local_input,
          out,
          dev_error,
          rank,
          world_size,
          n_elems,
          signal_token,
          options.wg_size,
          options.max_blocks);
      signal_token += 2;
      q_->wait();
      int host_error = 0;
      q_->memcpy(&host_error, dev_error, sizeof(int)).wait();
      if (host_error != 0) {
        throw std::runtime_error("One-shot allreduce warmup failed (device sync timeout).");
      }
    }

    std::cout << "[rank " << rank << "] Starting one-shot allreduce benchmark for "
              << options.iterations << " iterations..." << std::endl;
    q_->memset(dev_error, 0, sizeof(int)).wait();
     MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < options.iterations; ++iter) {
      launch_one_shot_allreduce<NUM_PER_TH>(
          dev_slot_ptrs,
          symm_->remote_signal_ptrs_dev_,
          local_slot,
          local_input,
          out,
          dev_error,
          rank,
          world_size,
          n_elems,
          signal_token,
          options.wg_size,
          options.max_blocks);
      signal_token += 2;
      MPI_Barrier(MPI_COMM_WORLD);
    }
    q_->wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[rank " << rank << "] One-shot allreduce benchmark completed." << std::endl;

    int host_error = 0;
    q_->memcpy(&host_error, dev_error, sizeof(int)).wait();
    if (host_error != 0) {
      throw std::runtime_error("One-shot allreduce failed (device sync timeout).");
    }

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / std::max(1, options.iterations);

    if (true) {
      std::cout << "Allreduce one-shot standalone: m=" << m
                << " n=" << n
                << " world_size=" << world_size
                << " avg_ms=" << avg_ms << std::endl;
    }

    bool passed = true;
    if (options.verify != 0) {
      // GPU result
      std::vector<float> host_gpu(static_cast<size_t>(n_elems));
      {
        std::vector<Element> tmp(static_cast<size_t>(n_elems));
        q_->memcpy(tmp.data(), out, static_cast<size_t>(n_elems) * sizeof(Element)).wait();
        for (size_t i = 0; i < tmp.size(); ++i) {
          host_gpu[i] = static_cast<float>(tmp[i]);
        }
      }

      // Reference: MPI_Allreduce of each rank's local data
      std::vector<float> host_local(static_cast<size_t>(n_elems),
                                    static_cast<float>(rank + 1));
      std::vector<float> host_ref(static_cast<size_t>(n_elems));
      MPI_Allreduce(host_local.data(), host_ref.data(),
                   static_cast<int>(n_elems), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

      size_t mismatch = 0;
      double max_abs = 0.0;
      double max_rel = 0.0;
      for (size_t i = 0; i < host_ref.size(); ++i) {
        double ref = static_cast<double>(host_ref[i]);
        double got = static_cast<double>(host_gpu[i]);
        double diff = std::abs(ref - got);
        double rel = (std::abs(ref) > 1e-6) ? diff / std::abs(ref) : diff;
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, rel);
        if (rel > 1e-2) ++mismatch;
      }

      passed = (mismatch == 0);
      std::cout << "[rank " << rank << "] verify "
                << (passed ? "PASSED" : "FAILED")
                << " ref[0]=" << host_ref[0]
                << " gpu[0]=" << host_gpu[0]
                << " max_abs=" << max_abs
                << " max_rel=" << max_rel
                << " mismatch=" << mismatch << "/" << host_ref.size()
                << std::endl;
    }

    // ─── RMS Norm ────────────────────────────────────────────────────────────
    constexpr float kRmsEps = 1e-6f;

    Element* weight  = sycl::malloc_device<Element>(static_cast<size_t>(n), *q_);
    Element* rms_out = sycl::malloc_device<Element>(static_cast<size_t>(n_elems), *q_);
    if (weight == nullptr || rms_out == nullptr) {
      throw std::runtime_error("Failed to allocate weight/rms_out buffers.");
    }
    // Initialize weight to 1.0 (identity scaling).
    q_->submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
        weight[i] = static_cast<Element>(1.0f);
      });
    }).wait();

    std::cout << "[rank " << rank << "] Starting RMS norm warmup..." << std::endl;
    for (int i = 0; i < 5; ++i) {
      launch_rms_norm(out, weight, rms_out, m, n, kRmsEps, options.wg_size);
    }
    q_->wait();

    std::cout << "[rank " << rank << "] Starting RMS norm benchmark for "
              << options.iterations << " iterations..." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < options.iterations; ++iter) {
      launch_rms_norm(out, weight, rms_out, m, n, kRmsEps, options.wg_size);
    }
    q_->wait();
    auto t3 = std::chrono::high_resolution_clock::now();

    double rms_total_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double rms_avg_ms   = rms_total_ms / std::max(1, options.iterations);
    std::cout << "RMS norm standalone: m=" << m
              << " n=" << n
              << " avg_ms=" << rms_avg_ms << std::endl;

    if (options.verify != 0) {
      std::vector<Element> h_in(static_cast<size_t>(n_elems));
      std::vector<Element> h_rms(static_cast<size_t>(n_elems));
      q_->memcpy(h_in.data(),  out,     static_cast<size_t>(n_elems) * sizeof(Element)).wait();
      q_->memcpy(h_rms.data(), rms_out, static_cast<size_t>(n_elems) * sizeof(Element)).wait();

      size_t rms_mismatch = 0;
      double rms_max_abs  = 0.0;
      double rms_max_rel  = 0.0;
      for (int row = 0; row < m; ++row) {
        double ss = 0.0;
        for (int col = 0; col < n; ++col) {
          double x = static_cast<double>(static_cast<float>(h_in[row * n + col]));
          ss += x * x;
        }
        double rms_inv = 1.0 / std::sqrt(ss / static_cast<double>(n) +
                                         static_cast<double>(kRmsEps));
        for (int col = 0; col < n; ++col) {
          double x   = static_cast<double>(static_cast<float>(h_in[row * n + col]));
          double ref = x * rms_inv; // weight == 1
          double got = static_cast<double>(static_cast<float>(h_rms[row * n + col]));
          double diff = std::abs(ref - got);
          double rel  = (std::abs(ref) > 1e-6) ? diff / std::abs(ref) : diff;
          rms_max_abs = std::max(rms_max_abs, diff);
          rms_max_rel = std::max(rms_max_rel, rel);
          if (rel > 1e-2) ++rms_mismatch;
        }
      }
      bool rms_passed = (rms_mismatch == 0);
      passed = passed && rms_passed;
      std::cout << "[rank " << rank << "] RMS norm verify "
                << (rms_passed ? "PASSED" : "FAILED")
                << " max_abs=" << rms_max_abs
                << " max_rel=" << rms_max_rel
                << " mismatch=" << rms_mismatch << "/" << n_elems
                << std::endl;
    }

    sycl::free(weight,  *q_);
    sycl::free(rms_out, *q_);

    sycl::free(dev_slot_ptrs, *q_);
    sycl::free(local_input, *q_);
    sycl::free(out, *q_);
    sycl::free(dev_error, *q_);
    return passed;
  }
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    if (rank == 0) {
      options.print_usage(std::cout) << std::endl;
    }
    MPI_Finalize();
    return 0;
  }

  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (devices.empty()) {
    if (rank == 0) {
      std::cerr << "No GPU devices found" << std::endl;
    }
    MPI_Finalize();
    return 1;
  }
  if (static_cast<size_t>(rank) >= devices.size()) {
    std::cerr << "Rank " << rank << " requires GPU device[" << rank
              << "], but only " << devices.size() << " devices are available" << std::endl;
    MPI_Finalize();
    return 1;
  }

  bool ok = false;
  try {
    Runner runner;
    ok = runner.run(options, devices[rank], rank, world_size);
  } catch (std::exception const& e) {
    std::cerr << "[rank " << rank << "] " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return ok ? 0 : 1;
}
