/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include <mpi.h>
#include <sycl/sycl.hpp>

#include <cmath>
#include <iostream>
#include <vector>

#include "reducescatter.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (world_size < 2) {
    if (rank == 0) {
      std::cerr << "[ipc-check] need at least 2 ranks\n";
    }
    MPI_Finalize();
    return 1;
  }

  sycl::async_handler async_handler = [rank](sycl::exception_list exceptions) {
    for (auto const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (const sycl::exception& ex) {
        std::cerr << "[rank " << rank << "] [ipc-check async] " << ex.what() << "\n";
      }
    }
  };

  try {
    // Explicitly select device based on rank
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (rs_log_enabled()) {
      std::cout << "[rank " << rank << "] Found " << devices.size() << " GPU device(s):\n";
      for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << "  [" << i << "] " << devices[i].get_info<sycl::info::device::name>() << "\n";
      }
    }
    if (devices.empty()) {
      std::cerr << "[rank " << rank << "] No GPU devices found\n";
      MPI_Finalize();
      return 1;
    }
    auto device = devices[rank % devices.size()];
    if (rs_log_enabled()) {
      std::cout << "[rank " << rank << "] Selected device[" << (rank % devices.size()) << "]: " << device.get_info<sycl::info::device::name>() << "\n";
    }

    sycl::queue q(device, async_handler, {sycl::property::queue::in_order()});

    constexpr int kElems = 256;
    float* local_buf = sycl::malloc_device<float>(kElems, q);
    int* local_flag = sycl::malloc_device<int>(world_size, q);

    // Step 1: Each rank initializes its own buffer with a rank-specific, index-checkable
    // pattern: local_buf[i] = rank * 1000.0f + i, and sets its own flag slot.
    {
      float base = static_cast<float>(rank) * 1000.0f;
      float* lb = local_buf;
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(kElems), [=](sycl::id<1> idx) {
          lb[idx[0]] = base + static_cast<float>(idx[0]);
        });
      }).wait();
    }
    {
      int r = rank;
      int* lf = local_flag;
      int ws = world_size;
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(ws), [=](sycl::id<1> idx) {
          lf[idx[0]] = (static_cast<int>(idx[0]) == r) ? (r + 1) : 0;
        });
      }).wait();
    }

    if (rs_log_enabled()) {
      std::cout << "[rank " << rank << "] [ipc-check] local_buf=" << static_cast<void*>(local_buf)
                << " initialized with base=" << (rank * 1000.0f) << "\n";
    }

    // Step 2: Barrier — ensure all ranks have written their own buffers before IPC exchange.
    MPI_Barrier(MPI_COMM_WORLD);

    // Step 3: Exchange IPC handles so every rank can access every other rank's buffer.
    std::vector<void*> opened_buf_ptrs;
    std::vector<void*> opened_flag_ptrs;
    float** ipc_buf_ptrs = exchange_ipc_ptrs(local_buf, rank, world_size, q, opened_buf_ptrs);
    int** ipc_flag_ptrs = exchange_ipc_ptrs(local_flag, rank, world_size, q, opened_flag_ptrs);

    for (int i = 0; i < world_size; ++i) {
      if (i == rank) {
        continue;
      }
      if (ipc_buf_ptrs[i] == nullptr || ipc_flag_ptrs[i] == nullptr) {
        std::cerr << "[rank " << rank << "] [ipc-check] null IPC ptr for peer " << i
                  << ", buf=" << static_cast<void*>(ipc_buf_ptrs[i])
                  << ", flag=" << static_cast<void*>(ipc_flag_ptrs[i]) << "\n";
        MPI_Abort(MPI_COMM_WORLD, 6);
      }
    }

    // Step 4: GPU kernel reads from src rank's buffer via IPC and stores results locally.
    int src = (rank - 1 + world_size) % world_size;
    // Check 4 representative indices: 0, 1, mid, last
    constexpr int kCheckElems = 4;
    int* check_indices = sycl::malloc_shared<int>(kCheckElems, q);
    check_indices[0] = 0;
    check_indices[1] = 1;
    check_indices[2] = kElems / 2;
    check_indices[3] = kElems - 1;
    float* read_buf = sycl::malloc_shared<float>(kCheckElems, q);
    int* read_flag = sycl::malloc_shared<int>(1, q);

    {
      int s = src;
      int* ci = check_indices;
      q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
          float* remote_buf = ipc_buf_ptrs[s];
          int* remote_flag = ipc_flag_ptrs[s];
          read_buf[0] = remote_buf[ci[0]];
          read_buf[1] = remote_buf[ci[1]];
          read_buf[2] = remote_buf[ci[2]];
          read_buf[3] = remote_buf[ci[3]];
          read_flag[0] = remote_flag[s];
        });
      }).wait_and_throw();
    }

    // Step 5: Verify the IPC-read values match the pattern src rank wrote to its own buffer.
    float src_base = static_cast<float>(src) * 1000.0f;
    bool buf_ok = true;
    int first_bad_check = -1;
    float first_bad_got = 0.0f;
    float first_bad_expected = 0.0f;
    for (int c = 0; c < kCheckElems; ++c) {
      float expected = src_base + static_cast<float>(check_indices[c]);
      if (std::fabs(read_buf[c] - expected) > 1e-3f) {
        buf_ok = false;
        first_bad_check = check_indices[c];
        first_bad_got = read_buf[c];
        first_bad_expected = expected;
        break;
      }
    }
    bool flag_ok = (read_flag[0] == src + 1);
    bool ok = buf_ok && flag_ok;

    if (rs_log_enabled()) {
      std::cout << "[rank " << rank << "] [ipc-check] IPC read from src=" << src
                << ": buf[0]=" << read_buf[0] << "(exp " << (src_base + check_indices[0]) << ")"
                << " buf[1]=" << read_buf[1] << "(exp " << (src_base + check_indices[1]) << ")"
                << " buf[" << check_indices[2] << "]=" << read_buf[2]
                << "(exp " << (src_base + check_indices[2]) << ")"
                << " buf[" << check_indices[3] << "]=" << read_buf[3]
                << "(exp " << (src_base + check_indices[3]) << ")"
                << " flag=" << read_flag[0] << "(exp " << (src + 1) << ")";
      if (!buf_ok) {
        std::cout << " FIRST_MISMATCH idx=" << first_bad_check
                  << " got=" << first_bad_got << " exp=" << first_bad_expected;
      }
      std::cout << " status=" << (ok ? "PASS" : "FAIL") << "\n";
    }

    int ok_int = ok ? 1 : 0;
    int all_ok = 0;
    MPI_Allreduce(&ok_int, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    close_ipc_ptrs(q, opened_buf_ptrs);
    close_ipc_ptrs(q, opened_flag_ptrs);
    sycl::free(ipc_buf_ptrs, q);
    sycl::free(ipc_flag_ptrs, q);
    sycl::free(check_indices, q);
    sycl::free(read_buf, q);
    sycl::free(read_flag, q);
    sycl::free(local_buf, q);
    sycl::free(local_flag, q);

    if (all_ok != 1) {
      if (rank == 0) {
        std::cerr << "[ipc-check] FAIL: IPC read check mismatch\n";
      }
      MPI_Abort(MPI_COMM_WORLD, 3);
    }

    if (rank == 0) {
      std::cout << "[ipc-check] PASS on all ranks\n";
    }
  } catch (const sycl::exception& ex) {
    std::cerr << "[rank " << rank << "] [ipc-check sync] " << ex.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 4);
  } catch (const std::exception& ex) {
    std::cerr << "[rank " << rank << "] [ipc-check std] " << ex.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 5);
  }

  MPI_Finalize();
  return 0;
}
