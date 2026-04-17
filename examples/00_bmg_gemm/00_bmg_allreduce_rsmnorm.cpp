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
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Standalone BMG One-Shot Allreduce Example\\n\\n"
        << "  --m=<int>            rows (default 8192)\\n"
        << "  --n=<int>            cols (default 4096)\\n"
        << "  --iterations=<int>   benchmark iterations (default 20)\\n"
        << "  --verify=<int>       0/1 verify output (default 1)\\n\\n"
        << "Constraints:\\n"
        << "  - This one-shot path is vector-only. m * n must be divisible by NUM_PER_TH * 4\\n"
        << "    bf16 elements. With the current build-time setting NUM_PER_TH=4, m * n must be\\n"
        << "    divisible by 16.\\n\\n";
    return out;
  }
};

template <int NUM_PER_TH>
class StandaloneOneShotAllreduceKernelName;

struct Runner {
  std::unique_ptr<sycl::queue> q_;
  std::unique_ptr<SymmMemory> symm_;

  template <int NUM_PER_TH>
    void launch_one_shot_allreduce(
      void** ipc_data_ptrs,
      uint32_t** ipc_signal_ptrs,
      Element* local_slot, //local symm data
      Element const* local_input,
      Element* out,
      int* dev_error,
      int rank,
      int world_size,
      int64_t n_elems,
      uint32_t signal_token) {

    constexpr int BF16_PER_I64 = 4;
    constexpr int BF16_VEC = NUM_PER_TH * BF16_PER_I64;

    if ((n_elems % BF16_VEC) != 0) {
      throw std::runtime_error(
        "n_elems must be divisible by NUM_PER_TH * 4 for one-shot vector allreduce.");
    }

    int dev_max_wg = static_cast<int>(
        q_->get_device().get_info<sycl::info::device::max_work_group_size>());
    // int wg_size = std::max(64, std::min(dev_max_wg, 256));
    const int wg_size = static_cast<int>(q_->get_device().get_info<sycl::info::device::max_work_group_size>()) / BF16_VEC;

    int64_t vec_elems = n_elems / BF16_VEC;
    int64_t blocks = std::max<int64_t>(1, (vec_elems + wg_size - 1) / wg_size);
    blocks = std::min<int64_t>(blocks, 32);
    int64_t global_size = blocks * wg_size;

    q_->submit([&](sycl::handler& h) {
      h.parallel_for<StandaloneOneShotAllreduceKernelName<NUM_PER_TH>>(
          sycl::nd_range<1>(
              sycl::range<1>(static_cast<size_t>(global_size)),
              sycl::range<1>(static_cast<size_t>(wg_size))),
          [=](sycl::nd_item<1> item) {
            one_shot_allreduce_device<NUM_PER_TH>(
                item,
                ipc_data_ptrs,
                ipc_signal_ptrs,
                reinterpret_cast<SyclBF16*>(local_slot),
                reinterpret_cast<SyclBF16 const*>(local_input),
                reinterpret_cast<SyclBF16*>(out),
                rank,
                world_size,
                n_elems,
                dev_error,
                signal_token);
          });
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
    constexpr int NUM_PER_TH = 4;
    constexpr int BF16_PER_I64 = 4;
    constexpr int BF16_VEC = NUM_PER_TH * BF16_PER_I64;
    if ((n_elems % BF16_VEC) != 0) {
      throw std::runtime_error(
          "n_elems must be divisible by NUM_PER_TH * 4 for one-shot vector allreduce.");
    }
    int wg_size = static_cast<int>(
        q_->get_device().get_info<sycl::info::device::max_work_group_size>()) / BF16_VEC;
    if (wg_size <= 0) {
      throw std::runtime_error("Invalid wg_size for one-shot allreduce.");
    }
    int64_t vec_elems = n_elems / BF16_VEC;
    int64_t blocks = std::max<int64_t>(1, (vec_elems + wg_size - 1) / wg_size);
    blocks = std::min<int64_t>(blocks, 32);
    size_t signal_pad_elems = static_cast<size_t>(blocks) * static_cast<size_t>(world_size);

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
    std::vector<void*> host_slot_ptrs(world_size);
    for (int r = 0; r < world_size; ++r) {
      uint8_t* base = reinterpret_cast<uint8_t*>(symm_->remote_data_ptrs_[r]);
      host_slot_ptrs[r] = base + static_cast<size_t>(r) * per_rank_elems * sizeof(Element);
    }
    void** dev_slot_ptrs = sycl::malloc_device<void*>(static_cast<size_t>(world_size), *q_);
    if (dev_slot_ptrs == nullptr) {
      throw std::runtime_error("Failed to allocate dev_slot_ptrs.");
    }
    q_->memcpy(dev_slot_ptrs, host_slot_ptrs.data(),
               static_cast<size_t>(world_size) * sizeof(void*)).wait();

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
          signal_token);
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
          signal_token);
      signal_token += 2;
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
      double gb = static_cast<double>(n_elems * sizeof(Element) * (world_size + 1)) / 1e9;
      double gbps = gb / (avg_ms / 1000.0);
      std::cout << "Allreduce one-shot standalone: m=" << m
                << " n=" << n
                << " world_size=" << world_size
                << " avg_ms=" << avg_ms
                << " BW=" << gbps << " GB/s" << std::endl;
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
