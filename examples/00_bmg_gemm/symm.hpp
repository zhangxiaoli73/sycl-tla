#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/initialize_block.hpp"

#include "common/sycl_cute_common.hpp"

using namespace cute;

// MPI datatype helper
template <typename T> MPI_Datatype mpi_type();
template <> MPI_Datatype mpi_type<float>()  { return MPI_FLOAT; }
template <> MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }

// ---------------- Level Zero IPC Helpers ----------------

inline ze_ipc_mem_handle_t ze_get_ipc_handle(sycl::context const& ctx, void* ptr) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    ze_ipc_mem_handle_t handle = {};
    auto res = zeMemGetIpcHandle(ze_ctx, ptr, &handle);
    assert(res == ZE_RESULT_SUCCESS && "zeMemGetIpcHandle failed");
    return handle;
}

inline void* ze_open_ipc_handle(sycl::context const& ctx, sycl::device const& dev,
                                ze_ipc_mem_handle_t handle) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    void* ptr = nullptr;
    auto res = zeMemOpenIpcHandle(ze_ctx, ze_dev, handle, 0, &ptr);
    assert(res == ZE_RESULT_SUCCESS && "zeMemOpenIpcHandle failed");
    return ptr;
}

inline void ze_close_ipc_handle(sycl::context const& ctx, void* ptr) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    zeMemCloseIpcHandle(ze_ctx, ptr);
}

// Exchange a device pointer across all MPI ranks via Level Zero IPC.
// Returns an array of world_size device-accessible pointers (caller must free with sycl::free).
// opened_ptrs collects remote pointers that need ze_close_ipc_handle on cleanup.
template <typename T>
T** exchange_ipc_ptrs(T* local_ptr, int rank, int world_size,
                      sycl::queue& Q, std::vector<void*>& opened_ptrs) {
    auto ctx = Q.get_context();
    auto dev = Q.get_device();

    auto local_handle = ze_get_ipc_handle(ctx, local_ptr);

    std::vector<ze_ipc_mem_handle_t> all_handles(world_size);
    MPI_Allgather(&local_handle, sizeof(ze_ipc_mem_handle_t), MPI_BYTE,
                  all_handles.data(), sizeof(ze_ipc_mem_handle_t), MPI_BYTE,
                  MPI_COMM_WORLD);

    T** ptrs = sycl::malloc_shared<T*>(world_size, Q);
    for (int i = 0; i < world_size; i++) {
        if (i == rank) {
            ptrs[i] = local_ptr;
        } else {
            void* remote = ze_open_ipc_handle(ctx, dev, all_handles[i]);
            ptrs[i] = static_cast<T*>(remote);
            opened_ptrs.push_back(remote);
        }
    }
    return ptrs;
}

inline void close_ipc_ptrs(sycl::queue& Q, std::vector<void*>& opened_ptrs) {
    auto ctx = Q.get_context();
    for (auto p : opened_ptrs) {
        ze_close_ipc_handle(ctx, p);
    }
    opened_ptrs.clear();
}

class SymmMemory {

public:
    SymmMemory(int m, int n, int k, int rank, int world_size, sycl::queue &Q)
        : m(m), n(n), k(k), rank(rank), world_size(world_size), init_Q(Q) {
    }

    ~SymmMemory() = default;

    template <class MMAType>
    void initialize_fused_ipc(TensorC_t& C, MMAType mma, sycl::queue& Q) {
        TC* c_ptr = C.data().get();

        int world_size = this.get_world_size();
        int m = this.get_m();
        int n = this.get_n();
        int tile_m = int(get<0>(mma.tile_mnk()));
        int tile_n = int(get<1>(mma.tile_mnk()));
        int num_m_tiles = int(ceil_div(m, tile_m));
        int num_n_tiles = int(ceil_div(n, tile_n));
        int required_tiles = num_m_tiles * num_n_tiles;

        // If C pointer changed or tile shape changed, rebuild IPC resources.
        if (fused_ipc_initialized_ && (local_c_ptr_ != c_ptr || num_tiles_ != required_tiles)) {
            release_fused_ipc();
        }

        if (fused_ipc_initialized_) {
            return;
        }

        int rank = ar_op.get_rank();
        num_tiles_ = required_tiles;

        local_symm_flag = sycl::malloc_device<int>(num_tiles_ * world_size, Q);
        local_symm_data = sycl::malloc_device<int>(num_tiles_ * world_size, Q); // todo: size is not correct
        Q.memset(local_symm_flag, 0, sizeof(int) * num_tiles_ * world_size).wait();
        Q.memset(local_symm_data, 0, sizeof(int) * num_tiles_ * world_size).wait();

        exchange_ipc_ptrs(local_symm_flag, rank, world_size, Q, opened_c_ptrs_);
        exchange_ipc_ptrs(local_symm_data, rank, world_size, Q, opened_signal_ptrs_);

        fused_ipc_initialized_ = true;
    }

    void release_fused_ipc() {
        if (!fused_ipc_initialized_) {
            return;
        }

        close_ipc_ptrs(gemm_q, opened_data_ptrs_);
        close_ipc_ptrs(gemm_q, opened_flag_ptrs_);

        if (opened_data_ptrs_) sycl::free(opened_data_ptrs_, gemm_q);
        if (opened_flag_ptrs_) sycl::free(opened_flag_ptrs_, gemm_q);

        opened_data_ptrs_ = nullptr;
        opened_flag_ptrs_ = nullptr;
        num_tiles_ = 0;
        fused_ipc_initialized_ = false;
    }


    int get_m() const { return m; }
    int get_n() const { return n; }
    int get_k() const { return k; }
    int get_world_size() const { return world_size; }
    int get_rank() const { return rank; }
    int get_data_ptr(int rank ) { return opened_data_ptrs_[rank]; }
    int get_flag_ptr(int rank ) { return opened_flag_ptrs_[rank]; }

private:
    int m, k, n, world_size, rank;
    sycl::queue& init_Q;
    int fused_ipc_initialized_;
    std::vector<void*> opened_data_ptrs_;
    std::vector<void*> opened_flag_ptrs_;
};