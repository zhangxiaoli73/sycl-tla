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
    \brief Distributed GEMM kernel layer helpers — XPU/SYCL version.

    Replaces PTX inline-asm uncached loads (ld.global.cv) with SYCL
    atomic_ref loads using memory_order::acquire + memory_scope::system
    to achieve the same cross-device cache-bypass / visibility semantics
    on Intel XPU.
*/

#pragma once

#include "cutlass/cutlass.h"

#include <sycl/sycl.hpp>

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::distributed_xpu::kernel::detail {

// Load a 64-bit value bypassing local caches, with cross-device acquire
// semantics (equivalent to PTX ld.global.cv on CUDA).
CUTLASS_DEVICE
void ld_without_cache(uint64_t& val, void const* ptr) {
#if defined(__SYCL_DEVICE_ONLY__)
  sycl::atomic_ref<uint64_t,
                   sycl::memory_order::acq_rel,
                   sycl::memory_scope::system,
                   sycl::access::address_space::global_space>
      ref(*reinterpret_cast<uint64_t*>(const_cast<void*>(ptr)));
  val = ref.load(sycl::memory_order::acquire);
#else
  val = *reinterpret_cast<const uint64_t*>(ptr);
#endif
}

// Load a 32-bit value bypassing local caches, with cross-device acquire
// semantics.
CUTLASS_DEVICE
void ld_without_cache(uint32_t& val, void const* ptr) {
#if defined(__SYCL_DEVICE_ONLY__)
  sycl::atomic_ref<uint32_t,
                   sycl::memory_order::acq_rel,
                   sycl::memory_scope::system,
                   sycl::access::address_space::global_space>
      ref(*reinterpret_cast<uint32_t*>(const_cast<void*>(ptr)));
  val = ref.load(sycl::memory_order::acquire);
#else
  val = *reinterpret_cast<const uint32_t*>(ptr);
#endif
}

} // namespace cutlass::distributed_xpu::kernel::detail

///////////////////////////////////////////////////////////////////////////////
