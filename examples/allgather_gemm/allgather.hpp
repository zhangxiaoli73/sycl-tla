#pragma once

#include <vector>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

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
#include "common/ishmem_common.hpp"

constexpr int kMaxWorldSize = 32;
constexpr int kNumSignals = 64;
constexpr int32_t kLocalCopyThreadPerBlock = 128;

constexpr int SPLIT = 1;

inline void *
ptr_offset(void *ptr, size_t offset) {
  return static_cast<char *>(ptr) + offset;
}

using namespace cute;

size_t get_local_copy_max_block_num(size_t num_input, int32_t pack_size = 1) {
  size_t total_blocks =
      (num_input / pack_size + kLocalCopyThreadPerBlock - 1) / kLocalCopyThreadPerBlock + 2;
  return total_blocks;
}

template <
    typename TA,
    char LayoutKindA,
    typename TensorA_t,
    typename TensorBarrierBuffer_t,
    typename TensorSyncBuffer_t,
    typename TensorSyncPtrBuffer_t>
class AllGatherOp {

public:
    AllGatherOp(int m, int n, int k, int rank, int world_size, sycl::queue &Q): m(m), n(n), k(k), rank(rank), world_size(world_size), init_Q(Q) {
        initialize();
    }

    ~AllGatherOp() {
        if (input_ptrs.size() > rank) {
            free_ishmem(input_ptrs[rank]);
        }
        if (barrier_ptrs.size() > rank) {
            free_ishmem(barrier_ptrs[rank]);
        }
        if (sync_ptrs.size() > rank) {
            free_ishmem(sync_ptrs[rank]);
        }
        if (sync_ptrs_buffer.size() > 0) {
            free_usm(init_Q, sync_ptrs_buffer.data().get());
        }
        // if (usm_sync_buffers != nullptr) {
        //     free_usm(init_Q, usm_sync_buffers);
        // }
    }

    void initialize() {
        int max_m_dim = m * world_size;
        // input buffers
        std::tie(this->input_buffers, this->input_ptrs) = create_ishmem_tensors<TensorA_t, TA, LayoutKindA>(init_Q, {max_m_dim, k});
        this->input_buffer_ = this->input_buffers[rank];
    
        // barrier buffers
        std::vector<void *> temp_ptrs;
        std::tie(this->barrier_buffers, temp_ptrs) = create_ishmem_tensors<TensorBarrierBuffer_t, int32_t, '-'>(init_Q, {kNumSignals});
        for (int i = 0; i < world_size; ++i) {
            this->barrier_ptrs.push_back(static_cast<int32_t *>(temp_ptrs[i]));
        }
        this->barrier_buffer_ = this->barrier_buffers[rank];

        // counter buffer
        // this->counter_buffer = make_shared_usm_tensor<int32_t, '-'>(Q, {kNumSignals + 1}, true);

        // sync buffer
        int local_copy_block_num = get_local_copy_max_block_num(m * k);
        // ishmem_malloc used. so no need to set mem to zero
        std::tie(this->sync_buffers, temp_ptrs) = create_ishmem_tensors<TensorSyncBuffer_t, int32_t, '-'>(init_Q, {world_size * local_copy_block_num});
        // std::tie(this->sync_buffers, temp_ptrs) = create_usm_tensors<TensorSyncBuffer_t, int32_t, '-'>(init_Q, {world_size * local_copy_block_num});
        // cannot use cute api, otherwise segmentfault
        // cute::clear(this->sync_buffers[rank]);
        init_Q.memset(this->sync_buffers[rank].data().get(), world_size * sizeof(int32_t), 0);
        for (int i = 0; i < world_size; ++i) {
            this->sync_ptrs.push_back(static_cast<int32_t *>(temp_ptrs[i]));
            std::cout << "node: " << rank << ", sync_ptrs[" << i << "] = " << this->sync_ptrs[i] << std::endl;
        }
        int sync_ptrs_buffer_size = sizeof(int32_t *) * world_size;
        this->sync_ptrs_buffer = make_shared_usm_tensor_init<char, '-'>(init_Q, {sync_ptrs_buffer_size}, false);
        init_Q.memcpy(this->sync_ptrs_buffer.data().get(), this->sync_ptrs.data(), sync_ptrs_buffer_size).wait();
        for (int i = 0; i < world_size; ++i) {
            std::cout << "node: " << rank << ", sync_ptrs_buffer[" << i << "] = " << ((int32_t**)this->sync_ptrs_buffer.data().get())[i] << std::endl;
        }
        // usm pointer buffer for kernel invocation
        // this->usm_sync_buffers = sycl::malloc_shared<int32_t *>(kMaxWorldSize, init_Q);

        // init params for d2d copy kernel
        // for (int i = 0; i < world_size; ++i) {
        //     this->copy_param.input_ptrs[i] = this->input_ptrs[i];
        //     // copy_param.scale_ptrs[i] = this->input_scale_ptrs[i];
        //     this->copy_param.ag_barriers[i] = this->barrier_ptrs[i];
        // }

        // this->copy_param.has_scale = false;
        // this->copy_param.counter = (int32_t *)this->counter_buffer.data();
        // this->copy_param.world_size = world_size;
        // // copy_param.sub_world_size = topo_utils::topo_numa_local_world_size();
        // this->copy_param.sub_world_size = world_size;
        // this->copy_param.rank = rank;
        // this->copy_param.ag_signal = (int32_t *)(copy_param.counter + world_size);
    }

    // Store value with release fence (for put_signal)
    // Order: store first, then release fence to flush the store
    inline void store_release(int32_t* addr, int32_t val) {
      *addr = val;
      sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    }

    // Load value with acquire fence (for get_signal/wait_signal)
    // Order: acquire fence first, then load to see the latest value
    inline int32_t load_acquire(int32_t* addr, int target_rank) {
      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
      int32_t *wait_ptr = addr + target_rank;
      int32_t val = *wait_ptr;
      return val;
    }


    sycl::event barrier_async(sycl::queue& Q) {
        std::cout << "node" << rank <<", barrier_async 1" << std::endl;
        // split to 2 kernels - k1: local write local (by memset after data copy), k2: remote write local (by pass cache?)
        // then, pattern should be like :
        // writer end: flag init in local ->  data write to remote -> release fence -> flag write to remote
        // read end: flag read local (by pass cache) -> data read

        return Q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(world_size), sycl::range<1>(world_size)), [=](sycl::nd_item<1> item) {
                int target_rank = item.get_local_id(0);
                if (target_rank < world_size) {
                   if (target_rank == rank ) {
                       return;
                   }
                    int32_t *sync_buffer_dst = sync_buffer_ptr[target_rank] + rank;
                    while (true) {
                      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                      int32_t *sync_buffer_dst = sync_buffer_ptr[target_rank] + rank;
                      int32_t val = *sync_buffer_dst;
                      if (val == 0) {break;}
                    }
                    *sync_buffer_dst = 1;
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

                     // step2: wait for my own signal to be updated
                    int32_t *wait_ptr = sync_buffer_ptr[rank] + target_rank;
                    while (true) {
                      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                      int32_t *wait_ptr = sync_buffer_ptr[rank] + target_rank;
                      int32_t val = *wait_ptr;
                      if (val == 1) {break;}
                    }
                    *wait_ptr = 0;
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

                }
            });
        });
        std::cout << "node" << rank <<", barrier_async 2" << std::endl;
        // ishmem_barrier_all();
        // return sycl::event();
    }

    template <typename Tensor_t, typename Element_t, char LayoutKind>
    auto create_ishmem_tensors(sycl::queue& Q, const std::vector<int>& shape) {
        auto [local_raw_ptr, local_layout_a] = make_ishmem_raw_ptr_and_layout<Element_t, LayoutKind>(Q, shape);
        auto local_engine = make_gmem_ptr(static_cast<Element_t*>(local_raw_ptr));

        std::vector<Tensor_t> input_buffers;
        std::vector<void *> input_ptrs;
        input_buffers.reserve(world_size);
        input_ptrs.reserve(world_size);
        for (int i = 0; i < world_size; ++i) {
            if (rank == i) {
                input_buffers.push_back(make_tensor(local_engine, local_layout_a));
                input_ptrs.push_back(local_raw_ptr);
            } else {
                void *remote_raw_ptr = ishmem_ptr(local_raw_ptr, i);
                input_buffers.push_back(make_tensor(make_gmem_ptr(static_cast<Element_t*>(remote_raw_ptr)), local_layout_a));
                input_ptrs.push_back(remote_raw_ptr);
            }
        }
        return std::make_tuple(input_buffers, input_ptrs);   
    }

    // template <typename Tensor_t, typename Element_t, char LayoutKind>
    // auto create_usm_tensors(sycl::queue& Q, const std::vector<int>& shape) {
    //     std::vector<Tensor_t> input_buffers;
    //     std::vector<void *> input_ptrs;
    //     input_buffers.reserve(world_size);
    //     input_ptrs.reserve(world_size);
    //     for (int i = 0; i < world_size; ++i) {
    //         auto [local_raw_ptr, local_layout_a] = make_usm_raw_ptr_and_layout<Element_t, LayoutKind>(Q, shape);
    //         auto local_engine = make_gmem_ptr(static_cast<Element_t*>(local_raw_ptr));
    //         input_buffers.push_back(make_tensor(local_engine, local_layout_a));
    //         input_ptrs.push_back(local_raw_ptr);
    //     }
    //     return std::make_tuple(input_buffers, input_ptrs);
    // }

    sycl::event set_ready(sycl::queue& Q, int32_t *barrier_ptr, sycl::event& cpy_event) {
        return Q.submit([&](sycl::handler& cgh) {
            // cgh.depends_on(cpy_event);
            cgh.single_task([=]() {
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, 
                                            sycl::memory_scope::device, 
                                            sycl::access::address_space::global_space> atm(*barrier_ptr);
                atm = 1;
            });
        });
        // sycl::event e = Q.memset(barrier_ptr, 1, sizeof(int32_t), cpy_event);
        // sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        // return e;
    }

    // sycl::event reset_barrier(sycl::queue& Q, int32_t *barrier_ptrs, int num_signals) {
    //     return Q.parallel_for(sycl::nd_range<1>(sycl::range<1>(kMaxWorldSize), sycl::range<1>(kMaxWorldSize)), [=](sycl::nd_item<1> item) {
    //         int tid = item.get_local_id(0);
    //         for (int i = 0; i < num_signals; i+= kMaxWorldSize) {
    //             *(barrier_ptrs + i) = 0;
    //         }
    //     });
    // }

    void local_copy_tensor_to_shmem_and_sync(TensorA_t const& inputA, sycl::queue& Q) {
        // if (local_cpy_to_shmem_event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {
        //     std::cout << "local copy to shmem status: not complete" << std::endl;
        // } else {
        //     std::cout << "local copy to shmem status: complete" << std::endl;
        // }
        std::cout << "node" << rank <<", local copy to shmem 1" << std::endl;
            size_t chunk_size = inputA.size() * sizeof(TA);
            assert(chunk_size == sizeof(TA) * m * k);
            void *input_ptr = inputA.data().get();
            void *input_buffer_ptr = this->input_buffer_.data().get();
            // barrier_async(Q, world_size);
            std::cout << "node" << rank <<", local copy to shmem 2" << std::endl;
            local_cpy_to_shmem_event = Q.memcpy(ptr_offset(input_buffer_ptr, rank * chunk_size), input_ptr, chunk_size);
            std::cout << "node" << rank <<", local copy to shmem 3" << std::endl;
            Q.memset(this->barrier_buffer_.data().get(), 0, sizeof(int32_t) * kNumSignals); // reset barrier
            std::cout << "node" << rank <<", local copy to shmem 4" << std::endl;
            // reset_barrier(Q, this->barrier_buffer_.data().get(), kNumSignals);
            // set input buffer of my rank ready
            // no sycl equivalent for CUStreamWriteValue
            // has to be in queue to avoid out of order execution
            int32_t *barrier_ptr = this->barrier_ptrs[rank] + rank;
            // set_ready(Q, barrier_ptr, local_cpy_to_shmem_event);
            std::cout << "node" << rank <<", local copy to shmem 5" << std::endl;
            this->barrier_sync_event = barrier_async(Q);
            std::cout << "node" << rank <<", local copy to shmem 6" << std::endl;
            if (this->barrier_sync_event.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {
                std::cout << "local copy to shmem status: not complete" << std::endl;
            } else {
                std::cout << "local copy to shmem status: complete" << std::endl;
            }
        // }
    }

    void copy_all_to_all(TensorA_t const& inputA, sycl::queue& Q) {
        size_t chunk_size = inputA.size() * sizeof(TA);
        // size_t split_chunk_size = chunk_size / SPLIT;
        for (int i = rank + 1; i < (world_size + rank); ++i) {
            auto id = i % world_size;
            // no split for now
            // for (int j = 0; j < SPLIT; ++j) {
            // auto split_offset = j * split_chunk_size;
            sycl::event cpy_event = Q.memcpy(
                ptr_offset(this->input_ptrs[rank], id * chunk_size),
                ptr_offset(this->input_ptrs[id], id * chunk_size),
                chunk_size);
            //set ready
            set_ready(Q, this->barrier_ptrs[rank] + id, cpy_event);
            // }
        }
    }

    void run(TensorA_t const& inputA, sycl::queue& Q) {
        std::cout << "node: " << rank << " ag_op run 1" << std::endl;
        // only use copy engine for now
        // init flag
        Q.memset(this->barrier_buffer_.data().get(), 0, sizeof(int32_t) * kNumSignals); // reset barrier

        // step1: copy local buffer to local symm buffer
        size_t chunk_size = inputA.size() * sizeof(TA);
        assert(chunk_size == sizeof(TA) * m * k);
        void *input_ptr = inputA.data().get(); //local buffer
        void *input_buffer_ptr = this->input_buffer_.data().get(); // symm buffer
        local_cpy_to_shmem_event = Q.memcpy(ptr_offset(input_buffer_ptr, rank * chunk_size), input_ptr, chunk_size); // from local to remote

        // step2: copy remote to local symm buffer, ring copy
        for (int step = 1; step <= world_size; ++step) {
            int remote_rank = (step + rank) % world_size;
            void* remote_p2p_bufs = this->input_ptrs[remote_rank];
            void* local_p2p_bufs = this->input_ptrs[rank]);
            Q.memcpy(local_p2p_bufs, remote_p2p_bufs, chunk_size);

            // step3: tell gemm that my copy is done
            Q.submit([&](sycl::handler& cgh) {
                cgh.single_task([=]() {
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                    *barrier_ptr = 1; //todo: what's the barrier_ptr?
                });
            });
        }
        // step4: sync all after all copy done
        barrier_async(Q);
    }

    sycl::event& get_local_cpy_to_shmem_event() {
        return this->local_cpy_to_shmem_event;
    }

    TensorBarrierBuffer_t& get_local_barrier_buffer() {
        return this->barrier_buffer_;
    }

    TensorA_t& get_local_input_buffer() {
        return this->input_buffer_;
    }

    std::vector<TensorA_t>& get_input_buffers() {
        return this->input_buffers;
    }

    int get_m() {
        return m;
    }

    int get_n() {
        return n;
    }

    int get_k() {
        return k;
    }

    int get_world_size() {
        return world_size;
    }

    int get_rank() {
        return rank;
    }
private:
    int m, k, n, world_size, rank;
    // used for the cuda-ipc-barrier
    std::vector<TensorSyncBuffer_t> sync_buffers;
    std::vector<TensorA_t> input_buffers;
    // std::vector<Tensor> input_scale_buffers;
    std::vector<TensorBarrierBuffer_t> barrier_buffers;
    TensorA_t input_buffer_;
    // Tensor input_scale_buffer_;
    // Tensor output_buffer;
    TensorBarrierBuffer_t barrier_buffer_;
    // Tensor counter_buffer;
    TensorSyncPtrBuffer_t sync_ptrs_buffer;  // symetric memory for barrier all. only with atomic supported
    // Tensor ag_signal;

    std::vector<void *> input_ptrs;
    // std::vector<void *> input_scale_ptrs;
    std::vector<int32_t *> barrier_ptrs;
    std::vector<int32_t *> sync_ptrs;

    // int32_t **usm_sync_buffers; // for kernel invocation

    // AllGatherParams copy_param;

    sycl::event local_cpy_to_shmem_event; // for async copy to shmem

    sycl::event local_cpy_debug_event;
    sycl::event barrier_sync_event;

    sycl::queue& init_Q;
};