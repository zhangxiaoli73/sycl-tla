/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
  \file Distributed GEMM Device Adapter — XPU/SYCL version

  Manages per-stage parameter arrays, buffer spaces, and barrier pointers
  for a distributed GEMM.  On SYCL, the CUDA-graph-based launch strategy is
  replaced with sequential submissions to the caller-supplied SYCL queue:

    1. Full barrier kernel    — q.single_task (via launch_full_barrier)
    2. Memcpy + flag writes   — q.memcpy / q.fill (only for HasMemcpy schedules)
    3. Per-iteration GEMMs    — DeviceGemm::run (which submits SYCL kernels)

  Because SYCL in-order queues serialise submissions, no graph is needed.
  Kernels for iteration > 0 that need remote data will spin-wait inside the
  kernel itself (see kernel/dist_gemm_kernel_wrapper.hpp::barrier_buffer).

  The public API mirrors the CUDA DistributedGemmUniversalAdapter exactly
  (same argument and state types), so callers can be switched between the
  two adapters with a simple namespace alias.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/workspace.h"

#include "cutlass/experimental/distributed_xpu/device/full_barrier.hpp"
#include "cutlass/experimental/distributed_xpu/device/detail.hpp"

#include <sycl/sycl.hpp>
#include "cutlass/util/sycl_event_manager.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::distributed_xpu::device {

template <class GemmKernel_>
class DistributedGemmUniversalAdapter {
public:
  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_>;
  using GemmKernel = GemmKernel_;
  using TileShape  = typename GemmKernel::TileShape;

  using ElementA           = typename GemmKernel::ElementA;
  using ElementB           = typename GemmKernel::ElementB;
  using ElementC           = typename GemmKernel::ElementC;
  using ElementD           = typename GemmKernel::ElementD;
  using ElementAccumulator = typename GemmKernel::ElementAccumulator;
  using DispatchPolicy     = typename GemmKernel::DispatchPolicy;
  using CollectiveMainloop = typename GemmKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename GemmKernel::CollectiveEpilogue;

  using LayoutA = typename DeviceGemm::LayoutA;
  using LayoutB = typename DeviceGemm::LayoutB;
  using LayoutC = typename DeviceGemm::LayoutC;
  using LayoutD = typename DeviceGemm::LayoutD;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  using MathOperator        = typename DeviceGemm::MathOperator;
  using OperatorClass       = typename DeviceGemm::OperatorClass;
  using ArchTag             = typename DeviceGemm::ArchTag;
  using ThreadblockSwizzle  = typename DeviceGemm::ThreadblockSwizzle;
  using ThreadblockShape    = typename DeviceGemm::ThreadblockShape;
  using ClusterShape        = typename DeviceGemm::ClusterShape;
  using InstructionShape    = typename DeviceGemm::InstructionShape;
  using WarpCount           = typename DeviceGemm::WarpCount;
  using WarpShape           = typename DeviceGemm::WarpShape;
  using EpilogueOutputOp    = typename DeviceGemm::EpilogueOutputOp;

  static int const  kThreadCount      = DeviceGemm::kThreadCount;
  static int constexpr kStages        = DeviceGemm::kStages;
  static int constexpr kAlignmentA    = DeviceGemm::kAlignmentA;
  static int constexpr kAlignmentB    = DeviceGemm::kAlignmentB;
  static int constexpr kAlignmentC    = DeviceGemm::kAlignmentC;
  static int constexpr kAlignmentD    = DeviceGemm::kAlignmentD;
  static int constexpr kSplitKAlignment = DeviceGemm::kSplitKAlignment;

  // Distributed GEMM types
  using DistSchedule  = typename GemmKernel::DistSchedule;
  static constexpr bool HasMemcpy = DistSchedule::HasMemcpy;
  using TP        = typename DistSchedule::TP;
  static constexpr int TP_ = TP{};

  using ElementFlag    = typename GemmKernel::ElementFlag;
  using ElementBarrier = uint32_t;

  using BufferHelper = detail::DistGemmBufferHelper<
      DistSchedule, ElementA, ElementB, ElementC, ElementD>;

  // Argument types
  using Arguments          = typename GemmKernel::BaseArguments;
  using DistributedArguments = typename GemmKernel::DistributedArguments;
  using PackedArguments    = typename GemmKernel::PackedArguments;
  using Params             = typename GemmKernel::PackedParams;

  // -------------------------------------------------------------------------
  // Distributed GEMM runtime state
  //
  // Unlike the CUDA adapter, there is no CUDA graph.  The queue itself
  // serialises all submitted work.
  // -------------------------------------------------------------------------
  struct DistributedGemmState {
    int device_idx = 0;

    Params params_array[TP_];

    // Memcpy pointers (used only when DistSchedule::HasMemcpy == true)
    void *       memcpy_source_ptr_array[TP_];
    void const*  memcpy_remote_ptr_array[TP_];
    size_t       memcpy_bytes[TP_];

    cutlass::Array<ElementBarrier*, TP_> device_barrier_ptrs;

    bool is_initialized = false;
  };

private:
  DistributedGemmState state_;

public:

  bool is_initialized() const { return state_.is_initialized; }

  // -------------------------------------------------------------------------
  // Validation
  // -------------------------------------------------------------------------

  static Status
  can_implement(Arguments const& args) {
    if (args.epilogue.thread.beta != 0.0 && DistSchedule::RemoteC) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Selected TP uses Remote C which requires "
                         "sourceless epilogue (beta == 0).\n");
      return Status::kInvalid;
    }
    if (not DistSchedule::can_implement_global(args.problem_shape)) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem shape not divisible by TP.\n");
      return Status::kInvalid;
    }
    Arguments args_copy = args;
    args_copy.problem_shape = DistSchedule::get_local_gemm_shape(args.problem_shape);
    for (int iteration = 0; iteration < TP_; ++iteration) {
      if (not GemmKernel::can_implement(args_copy)) {
        return Status::kInvalid;
      }
    }
    return Status::kSuccess;
  }

  // -------------------------------------------------------------------------
  // Workspace / exclusive workspace size helpers
  // -------------------------------------------------------------------------

  static size_t
  get_buffer_space_size(Arguments const& args) {
    size_t buffer_bytes = BufferHelper::get_buffer_size(args.problem_shape);
    return round_nearest(buffer_bytes, MinWorkspaceAlignment);
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = get_buffer_space_size(args);
    for (int iteration = 0; iteration < TP_; ++iteration) {
      workspace_bytes += GemmKernel::get_workspace_size(args);
    }
    return workspace_bytes;
  }

  static size_t get_barrier_bytes() {
    return round_nearest(sizeof(ElementBarrier), 32);
  }

  static size_t get_flag_bytes() {
    return round_nearest(sizeof(ElementFlag) * TP_, 32);
  }

  static size_t get_exclusive_workspace_size() {
    return get_barrier_bytes() + get_flag_bytes();
  }

  static void*
  exclusive_workspace_ptr_to_flag_ptr(void* exclusive_workspace_ptr, int iteration) {
    return static_cast<void*>(
        static_cast<uint8_t*>(exclusive_workspace_ptr) +
        get_barrier_bytes() +
        (sizeof(ElementFlag) * iteration));
  }

  // -------------------------------------------------------------------------
  // Tensor helpers — identical to the CUDA version
  // -------------------------------------------------------------------------

  static auto
  get_tensor_A_for_iter(Arguments const* args_array, void** buffer_space,
                        int device_idx, int iteration) {
    auto args     = args_array[device_idx];
    auto tensor_A = make_tensor(args.mainloop.ptr_A,
        make_layout(DistSchedule::get_local_a_shape(args.problem_shape), args.mainloop.dA));

    uint8_t* tensor_buffer = reinterpret_cast<uint8_t*>(buffer_space[device_idx]) +
        BufferHelper::get_buffer_offset_A(args.problem_shape);

    return DistSchedule::get_tensor_A(tensor_A, tensor_buffer, device_idx, iteration);
  }

  static auto
  get_tensor_B_for_iter(Arguments const* args_array, void** buffer_space,
                        int device_idx, int iteration) {
    auto args     = args_array[device_idx];
    auto tensor_B = make_tensor(args.mainloop.ptr_B,
        make_layout(DistSchedule::get_local_b_shape(args.problem_shape), args.mainloop.dB));

    uint8_t* tensor_buffer = reinterpret_cast<uint8_t*>(buffer_space[device_idx]) +
        BufferHelper::get_buffer_offset_B(args.problem_shape);

    return DistSchedule::get_tensor_B(tensor_B, tensor_buffer, device_idx, iteration);
  }

  static auto
  get_tensor_C_for_iter(Arguments const* args_array, void** buffer_space,
                        int device_idx, int iteration) {
    auto args     = args_array[device_idx];
    auto tensor_C = make_tensor(args.epilogue.ptr_C,
        make_layout(DistSchedule::get_local_c_shape(args.problem_shape), args.epilogue.dC));

    auto peer_idx_iter = DistSchedule::get_remote_peer_id(device_idx, iteration);
    void* buffer_ptr   = DistSchedule::RemoteC
                           ? buffer_space[peer_idx_iter]
                           : buffer_space[device_idx];

    uint8_t* tensor_buffer = reinterpret_cast<uint8_t*>(buffer_ptr) +
        BufferHelper::get_buffer_offset_C(args.problem_shape);

    return DistSchedule::get_tensor_C(tensor_C, tensor_buffer, device_idx, iteration);
  }

  static auto
  get_tensor_D_for_iter(Arguments const* args_array, void** buffer_space,
                        int device_idx, int iteration) {
    auto args     = args_array[device_idx];
    auto tensor_D = make_tensor(args.epilogue.ptr_D,
        make_layout(DistSchedule::get_local_d_shape(args.problem_shape), args.epilogue.dD));

    uint8_t* tensor_buffer = reinterpret_cast<uint8_t*>(buffer_space[device_idx]) +
        BufferHelper::get_buffer_offset_D(args.problem_shape);

    return DistSchedule::get_tensor_D(tensor_D, tensor_buffer, device_idx, iteration);
  }

  // -------------------------------------------------------------------------
  // Initialize
  //
  // Computes per-iteration argument/param structs and stores them in state_.
  // No SYCL graph is constructed — the queue in run() serialises all work.
  // -------------------------------------------------------------------------
  Status
  initialize(
      Arguments const* args,
      void**           workspace_ptrs,
      void**           exclusive_workspace_ptrs,
      int              device_idx,
      cudaStream_t     stream = nullptr)
  {
    CUTLASS_TRACE_HOST("DistributedGemmXpu::initialize() - stream: "
                       << (stream ? "non-null" : "null"));

    state_.device_idx = device_idx;

    for (int device = 0; device < TP_; ++device) {
      state_.device_barrier_ptrs[device] =
          reinterpret_cast<ElementBarrier*>(exclusive_workspace_ptrs[device]);
    }

    // Zero the exclusive workspace (barrier counter + flag array) for this device.
    Status status = zero_workspace(exclusive_workspace_ptrs[device_idx],
                                   get_exclusive_workspace_size(), stream, nullptr);
    if (status != Status::kSuccess) {
      return status;
    }

    void** buffer_space = workspace_ptrs;

    for (int iteration = 0; iteration < TP_; ++iteration) {
      size_t   workspace_iter_offset = GemmKernel::get_workspace_size(args[device_idx]);
      uint8_t* workspace_ptr =
          reinterpret_cast<uint8_t*>(workspace_ptrs[device_idx]) +
          get_buffer_space_size(args[device_idx]) +
          (iteration * workspace_iter_offset);

      void* workspace_iter = reinterpret_cast<void*>(workspace_ptr);

      // Build per-iteration tensor slices
      auto tensor_a_iter = get_tensor_A_for_iter(args, buffer_space, device_idx, iteration);
      auto tensor_b_iter = get_tensor_B_for_iter(args, buffer_space, device_idx, iteration);
      auto tensor_c_iter = get_tensor_C_for_iter(args, buffer_space, device_idx, iteration);
      auto tensor_d_iter = get_tensor_D_for_iter(args, buffer_space, device_idx, iteration);

      Arguments base_args      = args[device_idx];
      base_args.problem_shape  = DistSchedule::get_local_gemm_shape(args[device_idx].problem_shape);
      base_args.mainloop = {
          reinterpret_cast<const ElementA*>(tensor_a_iter.data()),
          tensor_a_iter.stride(),
          reinterpret_cast<const ElementB*>(tensor_b_iter.data()),
          tensor_b_iter.stride()
      };
      base_args.epilogue = {
          base_args.epilogue.thread,
          reinterpret_cast<const ElementC*>(tensor_c_iter.data()),
          tensor_c_iter.stride(),
          reinterpret_cast<ElementD*>(tensor_d_iter.data()),
          tensor_d_iter.stride()
      };

      if constexpr (DistSchedule::RemoteC) {
        base_args.epilogue.thread.beta = (iteration == 0) ? 0.0f : 1.0f;
      }

      auto [left_peer_idx, right_peer_idx] = DistSchedule::get_peers_for_device(device_idx);
      int flag_peer_idx = DistSchedule::KernelWritesArrivalFlag ? right_peer_idx : device_idx;

      void* self_flag_ptr =
          exclusive_workspace_ptr_to_flag_ptr(exclusive_workspace_ptrs[device_idx], iteration);
      void* peer_flag_ptr =
          exclusive_workspace_ptr_to_flag_ptr(exclusive_workspace_ptrs[flag_peer_idx], iteration);

      DistributedArguments distributed_args = {
          device_idx, iteration, self_flag_ptr, peer_flag_ptr};
      PackedArguments args_iter = {base_args, distributed_args};

      // Initialise workspace for this iteration
      status = GemmKernel::initialize_workspace(args_iter, workspace_iter, stream);
      if (status != Status::kSuccess) {
        return status;
      }

      // Store per-iteration params
      state_.params_array[iteration] =
          GemmKernel::to_underlying_arguments(args_iter, workspace_iter);

      // Record memcpy pointers (HasMemcpy schedules, iterations 1..TP-1)
      if (iteration > 0 && HasMemcpy) {
        auto peer_idx_iter = DistSchedule::get_remote_peer_id(device_idx, iteration);

        void*       local_ptr  = nullptr;
        void const* remote_ptr = nullptr;
        size_t      copy_size  = 0;

        static_assert(not DistSchedule::HasMemcpy || (DistSchedule::MemcpyA || DistSchedule::MemcpyB),
            "Expected to copy either A or B for memcpy schedules.");

        if constexpr (DistSchedule::MemcpyA) {
          copy_size  = cute::cosize(tensor_a_iter.layout()) * sizeof(ElementA);
          local_ptr  = reinterpret_cast<void*>(tensor_a_iter.data());
          auto remote_tensor = get_tensor_A_for_iter(args, buffer_space, peer_idx_iter, 0);
          remote_ptr = reinterpret_cast<void const*>(remote_tensor.data());
          assert(copy_size == cute::cosize(remote_tensor.layout()) * sizeof(ElementA));
        } else if constexpr (DistSchedule::MemcpyB) {
          copy_size  = cute::cosize(tensor_b_iter.layout()) * sizeof(ElementB);
          local_ptr  = reinterpret_cast<void*>(tensor_b_iter.data());
          auto remote_tensor = get_tensor_B_for_iter(args, buffer_space, peer_idx_iter, 0);
          remote_ptr = reinterpret_cast<void const*>(remote_tensor.data());
          assert(copy_size == cute::cosize(remote_tensor.layout()) * sizeof(ElementB));
        }

        state_.memcpy_source_ptr_array[iteration] = local_ptr;
        state_.memcpy_remote_ptr_array[iteration] = remote_ptr;
        state_.memcpy_bytes[iteration]            = copy_size;
      }
    } // for iteration

    state_.is_initialized = true;
    return Status::kSuccess;
  }

  Status
  update(Arguments const& /*args*/, void* /*workspace*/ = nullptr) {
    CUTLASS_TRACE_HOST("  DistributedGemmXpu does not support updating arguments yet.");
    return Status::kErrorInternal;
  }

  // -------------------------------------------------------------------------
  // run() — sequential SYCL queue submission (no SYCL graph)
  //
  // Execution order on the in-order queue:
  //   1. full_barrier kernel           (single_task — global any-to-any sync)
  //   2. (HasMemcpy only)
  //        for iteration 1..TP-1:
  //          queue.memcpy  remote → local buffer
  //          queue.fill    peer_flag_ptr ← ~0  (signal arrival)
  //   3. for iteration 0..TP-1:
  //        DeviceGemm::run(params, stream)
  //          iteration 0 — starts immediately
  //          iteration i>0 — kernel spins in barrier_buffer() until flag is set
  //                          by a PEER GPU's GEMM kernel or memcpy fill above.
  // -------------------------------------------------------------------------
  static Status
  run(DistributedGemmState& state, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("DistributedGemmXpu::run()");

    if (not state.is_initialized) {
      CUTLASS_TRACE_HOST("  Not initialized — call initialize() first.");
      return Status::kErrorInternal;
    }

    sycl::queue q = stream ? *stream : compat::get_default_queue();

    // 1. Full device barrier
    cutlass::Array<ElementFlag*, TP_> self_flag_ptrs;
    for (int iter = 0; iter < TP_; ++iter) {
      self_flag_ptrs[iter] = state.params_array[iter].distributed.self_flag_ptr_;
    }
    launch_full_barrier<TP_, ElementBarrier, TP_, ElementFlag>(
        state.device_barrier_ptrs, self_flag_ptrs, state.device_idx, stream);

    // 2. (HasMemcpy) Copy peer slices + signal flags
    if constexpr (HasMemcpy) {
      for (int iter = 1; iter < TP_; ++iter) {
        // Copy remote peer's slice into local buffer
        q.memcpy(state.memcpy_source_ptr_array[iter],
                 state.memcpy_remote_ptr_array[iter],
                 state.memcpy_bytes[iter]);

        // Set flag to non-zero so the corresponding GEMM kernel can proceed.
        // Use fill with ~0 (same as the CUDA 0xFF memset).
        ElementFlag flag_val = ~ElementFlag(0);
        q.fill(reinterpret_cast<ElementFlag*>(
                   state.params_array[iter].distributed.peer_flag_ptr_),
               flag_val, size_t(1));
      }
    }

    // 3. Submit per-iteration GEMM kernels in order.
    //    Because the SYCL queue is in-order, each iteration starts only after
    //    the preceding one completes on this device.  Cross-device
    //    synchronisation is handled by the kernel-internal spin-wait
    //    (barrier_buffer in dist_gemm_kernel_wrapper.hpp).
    for (int iter = 0; iter < TP_; ++iter) {
      Status status = DeviceGemm::run(state.params_array[iter], stream);
      if (status != Status::kSuccess) {
        return status;
      }
    }

    return Status::kSuccess;
  }

  // -------------------------------------------------------------------------
  // Convenience non-static overloads
  // -------------------------------------------------------------------------

  Status run(cudaStream_t stream = nullptr) {
    return run(state_, stream);
  }

  Status operator()(cudaStream_t stream = nullptr) {
    return run(state_, stream);
  }

  Status
  run(Arguments const* args,
      void**           workspace_ptrs,
      void**           exclusive_workspace_ptrs,
      int              device_idx,
      cudaStream_t     stream = nullptr) {
    Status status = initialize(args, workspace_ptrs, exclusive_workspace_ptrs, device_idx, stream);
    if (status == Status::kSuccess) {
      status = run(stream);
    }
    return status;
  }

  Status
  operator()(Arguments const* args,
             void**           workspace_ptrs,
             void**           exclusive_workspace_ptrs,
             int              device_idx,
             cudaStream_t     stream = nullptr) {
    return run(args, workspace_ptrs, exclusive_workspace_ptrs, device_idx, stream);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::distributed_xpu::device

////////////////////////////////////////////////////////////////////////////////
