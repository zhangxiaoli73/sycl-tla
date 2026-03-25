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
  \file Distributed GEMM Kernel Wrapper — XPU/SYCL version

  Wraps a CUTLASS 3.x GEMM kernel with the distributed-GEMM preamble:
    - Optionally signals an arrival flag on the left-peer GPU after the
      preceding GEMM completes (KernelWritesArrivalFlag schedules).
    - Spin-waits for a flag written by the right-peer GPU before starting
      the local GEMM for iteration > 0 (barrier_buffer).
    - Launches the underlying base kernel.

  SYCL adaptations vs the CUDA version:
    - arch::launch_dependent_grids() / wait_on_dependent_grids() are
      no-ops on XPU architectures; kept for structural symmetry.
    - blockIdx / threadIdx are not available in SYCL device code.
      maybe_signal_arrival() uses sycl::nd_item::get_group_linear_id()
      and get_local_linear_id() instead.
    - The flag write uses sycl::atomic_ref with memory_order::release +
      memory_scope::system to ensure cross-device visibility.
    - __nanosleep(40) is removed; the spin loop is a plain busy-wait.
    - cudaStream_t = sycl::queue* (via cutlass/gpu_generics.h alias).
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/experimental/distributed_xpu/kernel/detail.hpp"

#include <sycl/sycl.hpp>

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::distributed_xpu::kernel {

namespace detail {

// Allow all CUTLASS 3.X GEMM kernels
template <typename GemmKernel_>
struct SupportsDistributedGemm: cutlass::gemm::detail::IsCutlass3GemmKernel<GemmKernel_> {};

} // namespace detail

/*!
  DistributedGemmKernelWrapper — XPU/SYCL variant.

  Prepends the local GEMM with arrival-flag signalling and buffer barrier
  logic adapted for Intel XPU / SYCL compilation.
*/
template <class GemmKernel_, class DistSchedule_, class Enable = void>
struct DistributedGemmKernelWrapper;

template <class GemmKernel_, class DistSchedule_>
struct DistributedGemmKernelWrapper<
  GemmKernel_,
  DistSchedule_,
  cute::enable_if_t<detail::SupportsDistributedGemm<GemmKernel_>::value>
  >: GemmKernel_
{
  using DistSchedule = DistSchedule_;
  using TP = typename DistSchedule::TP;

  static constexpr bool KernelWritesArrivalFlag = DistSchedule::KernelWritesArrivalFlag;

  using BaseKernel    = GemmKernel_;
  using BaseArguments = typename BaseKernel::Arguments;
  using BaseParams    = typename BaseKernel::Params;

  static_assert(not cute::is_same_v<typename BaseKernel::ElementC, void>,
      "DistributedGEMM epilogues must have a source.");

  using ElementFlag = uint32_t;

  // -------------------------------------------------------------------------
  // Device-side argument and parameter structs
  // -------------------------------------------------------------------------

  struct DistributedArguments {
    int   device_idx  = 0;
    int   iteration   = 0;
    void* self_flag_ptr{nullptr};
    void* peer_flag_ptr{nullptr};
  };

  struct PackedArguments {
    BaseArguments        base{};
    DistributedArguments distributed{};
  };

  struct DistributedParams {
    int          device_idx       = 0;
    int          iteration        = 0;
    ElementFlag* self_flag_ptr_   = nullptr;
    ElementFlag* peer_flag_ptr_   = nullptr;
  };

  struct PackedParams {
    BaseParams        base{};
    DistributedParams distributed{};
  };

  using Params = PackedParams;

  // -------------------------------------------------------------------------
  // Static helpers
  // -------------------------------------------------------------------------

  static PackedParams
  to_underlying_arguments(PackedArguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("distributed_xpu::to_underlying_arguments():");

    auto kernel_params = BaseKernel::to_underlying_arguments(args.base, workspace);

    DistributedParams dist_params = {
        args.distributed.device_idx,
        args.distributed.iteration,
        reinterpret_cast<ElementFlag*>(args.distributed.self_flag_ptr),
        reinterpret_cast<ElementFlag*>(args.distributed.peer_flag_ptr)
    };

    return {kernel_params, dist_params};
  }

  static bool
  can_implement(BaseArguments const& args) {
    return BaseKernel::can_implement(args);
  }

  static bool
  can_implement(PackedArguments const& args) {
    return BaseKernel::can_implement(args.base);
  }

  static size_t
  get_workspace_size(BaseArguments const& args) {
    return BaseKernel::get_workspace_size(args);
  }

  static size_t
  get_workspace_size(PackedArguments const& args) {
    return BaseKernel::get_workspace_size(args.base);
  }

  static cutlass::Status
  initialize_workspace(BaseArguments const& args,
                       void*        workspace   = nullptr,
                       cudaStream_t stream      = nullptr,
                       CudaHostAdapter* /*adapter*/ = nullptr) {
    return BaseKernel::initialize_workspace(args, workspace, stream, nullptr);
  }

  static cutlass::Status
  initialize_workspace(PackedArguments const& args,
                       void*        workspace   = nullptr,
                       cudaStream_t stream      = nullptr,
                       CudaHostAdapter* /*adapter*/ = nullptr) {
    return BaseKernel::initialize_workspace(args.base, workspace, stream, nullptr);
  }

  static dim3
  get_grid_shape(PackedParams const& params) {
    return BaseKernel::get_grid_shape(params.base);
  }

  static dim3
  get_grid_shape(BaseParams const& params) {
    return BaseKernel::get_grid_shape(params);
  }

  // -------------------------------------------------------------------------
  // Device-side preamble functions
  // -------------------------------------------------------------------------

  // Spin-wait on the self_flag written by the hosting device's memcpy/arrival
  // path.  Only called for iteration > 0.
  CUTLASS_DEVICE
  void
  barrier_buffer(PackedParams const& params) {
    if (params.distributed.iteration > 0) {
      using detail::ld_without_cache;

      ElementFlag comm_iter = 0;
      ld_without_cache(comm_iter, params.distributed.self_flag_ptr_);
      while (comm_iter == 0) {
        ld_without_cache(comm_iter, params.distributed.self_flag_ptr_);
        // No __nanosleep on XPU — plain busy-wait is acceptable here.
      }
    }
  }

  // Write the peer's arrival flag (1) from the first work-item of the first
  // work-group.  Uses a release-scoped atomic store so the write is visible
  // to the peer's cross-device spin-wait.
  CUTLASS_DEVICE
  void
  maybe_signal_arrival(PackedParams const& params) {
    if constexpr (KernelWritesArrivalFlag) {
#if defined(__SYCL_DEVICE_ONLY__)
      auto item = compat::get_nd_item<3>();
      bool is_first = (item.get_group_linear_id() == 0) &&
                      (item.get_local_linear_id()  == 0) &&
                      (params.distributed.iteration > 0);
      if (is_first) {
        sycl::atomic_ref<ElementFlag,
                         sycl::memory_order::release,
                         sycl::memory_scope::system,
                         sycl::access::address_space::global_space>
            flag_ref(*reinterpret_cast<ElementFlag*>(params.distributed.peer_flag_ptr_));
        flag_ref.store(ElementFlag(1));
      }
#endif
    }
  }

  // -------------------------------------------------------------------------
  // Kernel entry point
  // -------------------------------------------------------------------------

  CUTLASS_DEVICE
  void
  operator()(PackedParams const& params, char* smem_buf) {
    // On XPU, launch_dependent_grids / wait_on_dependent_grids are no-ops
    // (arch::IsGdcGloballyEnabled == false).  Kept for structural symmetry.
    arch::launch_dependent_grids();
    arch::wait_on_dependent_grids();

    // Optionally write arrival flag for the previous stage/iteration.
    maybe_signal_arrival(params);

    // Spin-wait until the local buffer is ready (filled by memcpy or peer GEMM).
    barrier_buffer(params);

    // Execute the local GEMM.
    BaseKernel gemm;
    gemm(params.base, smem_buf);
  }
};

} // namespace cutlass::distributed_xpu::kernel

///////////////////////////////////////////////////////////////////////////////
