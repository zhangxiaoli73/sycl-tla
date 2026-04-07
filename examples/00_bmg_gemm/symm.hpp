#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include <level_zero/ze_api.h>

#if defined(__linux__)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include "ipc.hpp"

#define ZE_CHECK(cmd) do {                             \
    ze_result_t e = (cmd);                             \
    if (e != ZE_RESULT_SUCCESS) {                      \
        printf("Level-Zero error at %s:%d code=%d\n", \
                __FILE__, __LINE__, e);                 \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

inline void ze_close_ipc_handle(sycl::context const& ctx, void* ptr) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    zeMemCloseIpcHandle(ze_ctx, ptr);
}

std::vector<void*> exchange_ipc_ptrs(void* ptr, int rank, int world_size, sycl::queue& q, std::vector<void*>& opened_ptrs) {

  auto ctx = q.get_context();
  auto dev = q.get_device();
  auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);

  // Step 1: Get base address and offset
  void* base_addr;
  size_t base_size;
  ZE_CHECK(zeMemGetAddressRange(l0_ctx, ptr, &base_addr, &base_size));
  size_t offset = (char*)ptr - (char*)base_addr;

  // Step 2: Get IPC mem handle from base address
  ze_ipc_mem_handle_t local_ipc_handle;
  ZE_CHECK(zeMemGetIpcHandle(l0_ctx, base_addr, &local_ipc_handle));

  // Step 3: Extract fd from IPC handle (ze_ipc_mem_handle_t's first field is fd)
  int local_fd = *reinterpret_cast<int*>(&local_ipc_handle);

  // Step 4: Exchange offsets via MPI
  std::vector<size_t> offsets(world_size);
  MPI_Allgather(&offset, sizeof(size_t), MPI_BYTE,
                offsets.data(), sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

  // Step 5: Gather PIDs from all ranks for fd exchange
  std::vector<int> pids(world_size);
  int local_pid = getpid();
  MPI_Allgather(&local_pid, 1, MPI_INT,
                pids.data(), 1, MPI_INT, MPI_COMM_WORLD);

  // Step 6: Exchange fds via IpcChannel (uses Unix domain socket + SCM_RIGHTS)
  IpcChannel ipc_channel;
  // Ensure all ranks have bound their sockets before anyone starts sending
  MPI_Barrier(MPI_COMM_WORLD);
  auto fds = ipc_channel.all_gather_fds(rank, pids, local_fd);

  // Step 7: Reconstruct remote IPC handles and open them
  std::vector<void*> buffers(world_size, nullptr);

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      buffers[r] = ptr;
      continue;
    }

    // Reconstruct remote IPC handle by setting the fd field
    ze_ipc_mem_handle_t remote_ipc_handle = local_ipc_handle; // Copy structure
    *reinterpret_cast<int*>(&remote_ipc_handle) = fds[r]; // Set remote fd

    // Open IPC handle to get remote base address
    void* remote_base;
    ZE_CHECK(zeMemOpenIpcHandle(
        l0_ctx,
        l0_device,
        remote_ipc_handle,
        ZE_IPC_MEMORY_FLAG_BIAS_CACHED,
        &remote_base));

    opened_ptrs.push_back(remote_base);
    buffers[r] = (char*)remote_base + offsets[r];
  }
  return buffers;
}

inline void close_ipc_ptrs(sycl::queue& q, std::vector<void*>& opened_ptrs) {
  auto ctx = q.get_context();
  for (void* p : opened_ptrs) {
    ze_close_ipc_handle(ctx, p);
  }
  opened_ptrs.clear();
}

// ---------------- Symmetric-memory style signal barrier ----------------

// This class provides barrier/put_signal/wait_signal semantics similar to
// XPUSymmetricMemory. Each rank owns one signal pad, IPC-opens all peer pads,
// and stores per-channel tickets in layout: [channel][src_rank].
class SymmMemory {
 public:
  SymmMemory(int m, int n, int k, int rank, int world_size, sycl::queue& q, int num_channels = 1024)
      : m_(m),
        n_(n),
        k_(k),
        rank_(rank),
        world_size_(world_size),
        num_channels_(num_channels),
        init_q_(q),
        local_epoch_(num_channels, 0) {
      // Data buffer layout for allgathered A shards: [world_size][m][k].
      // `m` here is local_m from caller, so total elements are world_size * local_m * k.
    size_t data_elems = static_cast<size_t>(m) * k * world_size_; // 16-bit elements
    size_t signal_elems = static_cast<size_t>(world_size_);

    size_t data_elems_bytes = data_elems * 2; // 16-bit elements
    size_t signal_elems_bytes = signal_elems * sizeof(uint32_t);

    std::cout << "zl_debug start to malloc local buffer and flag, "
              << "data_elems_bytes=" << data_elems_bytes
              << " signal_elems_bytes=" << signal_elems_bytes << std::endl;
    local_signal_ptr_ = sycl::malloc_device(signal_elems_bytes, init_q_);
    local_data_ptr_ = sycl::malloc_device(data_elems_bytes, init_q_);
    if (local_signal_ptr_ == nullptr || local_data_ptr_ == nullptr) {
      throw std::runtime_error("SymmMemory: sycl::malloc_device failed. signal_ptr="
          + std::to_string(reinterpret_cast<uintptr_t>(local_signal_ptr_))
          + " data_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(local_data_ptr_))
          + " signal_bytes=" + std::to_string(signal_elems_bytes)
          + " data_bytes=" + std::to_string(data_elems_bytes));
    }
    std::cout << "zl_debug finish malloc local buffer and flag " << std::endl;

    std::cout << "zl_debug memset signal buffer" << std::endl;
    init_q_.memset(local_signal_ptr_, 0, signal_elems_bytes).wait();
    std::cout << "zl_debug memset data buffer" << std::endl;
    init_q_.memset(local_data_ptr_, 0, data_elems_bytes).wait();
    std::cout << "zl_debug memset done" << std::endl;
    
    std::cout << "zl_debug start to do IPC exchange " << std::endl;
    remote_signal_ptrs_ = exchange_ipc_ptrs(local_signal_ptr_, rank_, world_size_, init_q_, opened_signal_bases_);
    remote_data_ptrs_ = exchange_ipc_ptrs(local_data_ptr_, rank_, world_size_, init_q_, opened_data_bases_);
    std::cout << "zl_debug finish the IPC exchange " << std::endl;

    // make remote IPC memory resident on local device
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(init_q_.get_context());
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(init_q_.get_device());

    std::cout << "zl_debug start to make resident" << std::endl;
    for (int peer = 0; peer < world_size_; ++peer) {
      if (peer == rank_) continue;
      if (remote_signal_ptrs_[peer] == nullptr) {
        throw std::runtime_error("SymmMemory remote_signal_ptr is null for peer " + std::to_string(peer));
      }
      auto res = zeContextMakeMemoryResident(ze_ctx, ze_dev, remote_signal_ptrs_[peer],
          signal_elems * sizeof(uint32_t));
      if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("zeContextMakeMemoryResident failed for remote signal ptr of peer " + std::to_string(peer));
      }
    }

    for (int peer = 0; peer < world_size_; ++peer) {
      if (peer == rank_) continue;
      if (remote_data_ptrs_[peer] == nullptr) {
        throw std::runtime_error("SymmMemory remote_data_ptr is null for peer " + std::to_string(peer));
      }
      auto res = zeContextMakeMemoryResident(ze_ctx, ze_dev, remote_data_ptrs_[peer],
          data_elems * sizeof(uint16_t));
      if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("zeContextMakeMemoryResident failed for remote data ptr of peer " + std::to_string(peer));
      }
    }
    std::cout << "zl_debug finish making resident" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  ~SymmMemory() {
    close_ipc_ptrs(init_q_, opened_signal_bases_);
    close_ipc_ptrs(init_q_, opened_data_bases_);

    remote_signal_ptrs_.clear();
    remote_data_ptrs_.clear();
    if (local_signal_ptr_) {
      sycl::free(local_signal_ptr_, init_q_);
      local_signal_ptr_ = nullptr;
    }
    if (local_data_ptr_) {
      sycl::free(local_data_ptr_, init_q_);
      local_data_ptr_ = nullptr;
    }
  }

  SymmMemory(SymmMemory const&) = delete;
  SymmMemory& operator=(SymmMemory const&) = delete;

  int get_m() const { return m_; }
  int get_n() const { return n_; }
  int get_k() const { return k_; }
  int get_rank() const { return rank_; }
  int get_world_size() const { return world_size_; }

  void check_channel(int channel) const {
    assert(channel >= 0 && "channel must be non-negative");
    assert(channel < num_channels_ && "channel exceeds configured num_channels");
  }

  // Equivalent semantics to XPUSymmetricMemory::barrier(channel):
  // - publish a ticket to peers' signal pads
  // - wait until every peer publishes the same ticket to this rank
  void barrier(int channel, size_t timeout_ms = 0) {
    barrier(channel, init_q_, timeout_ms);
  }

  void barrier(int channel, sycl::queue& queue, size_t timeout_ms = 0) {
    (void)timeout_ms;
    (void)channel;

    constexpr int barrier_channel = 0;

    uint32_t ticket = ++local_epoch_[barrier_channel];
    int rank = rank_;
    int world_size = world_size_;
    int base = barrier_channel * world_size_;
    std::cout << "zl_debug start to do memset " << std::endl;
    // memset to 0 before the first barrier
     for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) {
        continue;
      }
      uint32_t* remote_slot = reinterpret_cast<uint32_t*>(remote_signal_ptrs_[peer]);
      if (remote_slot == nullptr) {
        throw std::runtime_error("SymmMemory barrier remote_slot is null, unexpected.");
      }
      queue.memset(remote_slot, 0, world_size_ * sizeof(uint32_t)).wait();
    }

    std::cout << "zl_debug memset done" << std::endl;
    return;

//    queue.submit([&](sycl::handler& h) {
//      h.single_task([=]() {
//        // put_signal to all peers
//        for (int peer = 0; peer < world_size; ++peer) {
//          if (peer == rank) {
//            continue;
//          }
//          uint32_t* remote_slot = reinterpret_cast<uint32_t*>(pads[peer]) + base + rank;
//          remote_slot[0] = 0;
//          sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
//        }
//
//         wait_signal from all peers
//         uint32_t* my_pad = pads[rank] + base;
//         for (int peer = 0; peer < world_size; ++peer) {
//           if (peer == rank) {
//             continue;
//           }
//           while (true) {
//             sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
//             uint32_t* wait_slot = my_pad + peer;
//             if (*wait_slot >= ticket) {
//               break;
//             }
//           }
//         }
//      });
//    });
  }


  void* get_data_buffer(int rank) {
      return remote_data_ptrs_[rank];
  }

  void* get_flag_buffer(int rank) {
      return remote_signal_ptrs_[rank];
  }
  void* local_signal_ptr_ = nullptr;
  std::vector<void*> remote_signal_ptrs_;
  void* local_data_ptr_ = nullptr;
  std::vector<void*> remote_data_ptrs_;


 private:
  int m_;
  int n_;
  int k_;
  int rank_;
  int world_size_;
  int num_channels_;

  sycl::queue& init_q_;

  std::vector<void*> opened_signal_bases_;
  std::vector<void*> opened_data_bases_;
  std::vector<uint32_t> local_epoch_;
};
