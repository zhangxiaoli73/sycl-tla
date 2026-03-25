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
/*! \file
    \brief Distributed GEMM barrier kernel — XPU/SYCL version.

    This file provides full_barrier_kernel_impl(), a plain (non-__global__)
    device-callable function that is launched from device/full_barrier.hpp
    via a SYCL single_task kernel.  It replaces the CUDA __global__
    full_barrier_kernel which relied on atomicAdd/atomicSub PTX builtins
    and cudaGraphExec.

    Key differences from the CUDA version:
      - No CUTLASS_GLOBAL / __global__ qualifier.
      - sycl::atomic_ref<..., memory_scope::system> is used for all
        cross-device atomic operations and spin-reads to guarantee
        coherence across Intel XPU compute tiles.
      - No __nanosleep(40) — SYCL has no equivalent; the spin loop is a
        plain busy-wait which is acceptable inside a single_task kernel.
      - arch::launch_dependent_grids / wait_on_dependent_grids are not
        needed because the SYCL in-order queue enforces submission order.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#if defined(CUTLASS_ENABLE_SYCL)
#include <sycl/sycl.hpp>
#endif

#include "cutlass/experimental/distributed_xpu/kernel/detail.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::distributed_xpu::kernel {

/*!
  full_barrier_kernel_impl performs an any-to-any device barrier across NP
  participating devices.

  Steps:
    1. Reset per-stage arrival flags (local side-effect before incrementing
       peer counters — safe because this runs after the previous global sync).
    2. Atomically increment every peer's arrival counter by 1.
    3. Spin until our own arrival counter reaches NP-1.
    4. Atomically subtract NP-1 from our counter to reset it for reuse.

  Template parameters mirror the CUDA full_barrier_kernel:
    NP         — number of participating devices
    IntType    — type of the arrival counter (typically uint32_t)
    Iterations — number of per-stage flags to reset
    FlagType   — type of flag elements
*/
template <int NP, typename IntType, int Iterations, typename FlagType>
void full_barrier_kernel_impl(
    cutlass::Array<IntType*, NP>        device_arrival_ptrs,
    cutlass::Array<FlagType*, Iterations> iteration_flag_ptrs,
    IntType                              device_idx)
{
#if defined(__SYCL_DEVICE_ONLY__)

  // 1. Reset per-stage arrival flags before signalling peers.
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Iterations; ++i) {
    sycl::atomic_ref<FlagType,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        flag_ref(iteration_flag_ptrs[i][0]);
    flag_ref.store(static_cast<FlagType>(0));
  }

  // Ensure flag resets are visible to peers before we increment their counters.
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

  // 2. Atomically increment every peer's arrival counter.
  IntType val = IntType(1);
  CUTLASS_PRAGMA_UNROLL
  for (IntType d = 0; d < IntType(NP); ++d) {
    if (d != device_idx) {
      sycl::atomic_ref<IntType,
                       sycl::memory_order::relaxed,
                       sycl::memory_scope::system,
                       sycl::access::address_space::global_space>
          peer_ref(*device_arrival_ptrs[d]);
      peer_ref.fetch_add(val);
    }
  }

  // 3. Spin until our local arrival counter reaches NP-1.
  IntType max_val = static_cast<IntType>(NP - 1);
  sycl::atomic_ref<IntType,
                   sycl::memory_order::acquire,
                   sycl::memory_scope::system,
                   sycl::access::address_space::global_space>
      local_ref(*device_arrival_ptrs[device_idx]);

  IntType curr_val = local_ref.load();
  while (curr_val < max_val) {
    curr_val = local_ref.load();
  }

  // 4. Reset our local counter.
  sycl::atomic_ref<IntType,
                   sycl::memory_order::relaxed,
                   sycl::memory_scope::system,
                   sycl::access::address_space::global_space>
      reset_ref(*device_arrival_ptrs[device_idx]);
  reset_ref.fetch_sub(max_val);

#endif // __SYCL_DEVICE_ONLY__
}

} // namespace cutlass::distributed_xpu::kernel

///////////////////////////////////////////////////////////////////////////////
