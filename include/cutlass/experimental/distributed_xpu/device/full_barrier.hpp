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
    \brief Device layer interface for the Distributed GEMM barrier — XPU/SYCL version.

    Replaces the CUDA cudaGraphExec-based launch with a SYCL single_task submission
    to the caller-supplied (or default) sycl::queue.  The queue's in-order
    execution guarantee serialises the barrier after any preceding work and
    before the subsequent GEMM kernels — no SYCL graph is needed.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include "cutlass/experimental/distributed_xpu/kernel/full_barrier.hpp"

#if defined(CUTLASS_ENABLE_SYCL)
#include <sycl/sycl.hpp>
#include "cutlass/util/sycl_event_manager.hpp"
#endif

namespace cutlass::distributed_xpu::device {

/*!
  launch_full_barrier — XPU/SYCL version.

  Submits a single-work-item SYCL kernel to the provided queue (or the
  default queue if stream == nullptr) that performs the full any-to-any
  device barrier described in kernel/full_barrier.hpp.

  Parameters are identical to the CUDA version except that
    - cudaStream_t is sycl::queue* (via gpu_generics.h alias)
    - launch_with_pdl is absent (PDL is a CUDA-only concept)
*/
template <int NP, typename IntType, int Iterations, typename FlagType>
void launch_full_barrier(
    cutlass::Array<IntType*,   NP>         device_arrival_ptrs,
    cutlass::Array<FlagType*, Iterations>  iteration_flag_ptrs,
    IntType                                device_idx,
    cudaStream_t                           stream)
{
#if defined(CUTLASS_ENABLE_SYCL)
  sycl::queue q = stream ? *stream : compat::get_default_queue();

  auto e = q.single_task([=]() {
    cutlass::distributed_xpu::kernel::full_barrier_kernel_impl<
        NP, IntType, Iterations, FlagType>(
        device_arrival_ptrs,
        iteration_flag_ptrs,
        device_idx);
  });

  EventManager::getInstance().addEvent(e);
#endif // CUTLASS_ENABLE_SYCL
}

} // namespace cutlass::distributed_xpu::device
