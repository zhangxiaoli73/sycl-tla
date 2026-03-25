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
  \file Base Schedule for Distributed GEMM (XPU/SYCL version)

  Identical to the CUDA version in template logic; ported into the
  cutlass::distributed_xpu::schedules namespace so that device/kernel
  layers can reference it without pulling in the CUDA distributed headers.
*/

#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::distributed_xpu::schedules {

template <
  class TP_,
  class ProcessorTiler_,
  class IterationTiler_,
  class PeerDeviceMapping_,
  class IterationMappingM_,
  class IterationMappingN_,
  class IterationMappingK_,
  class IterationMappingL_,
  class ProcessorOffset_,
  bool MemcpyA_,
  bool MemcpyB_,
  bool KernelWritesArrivalFlag_,
  int NumBuffersA_,
  int NumBuffersB_,
  int NumBuffersC_,
  int NumBuffersD_>
struct BaseSchedule {

  using TP = TP_;

  static_assert(
      cute::is_static<TP>::value && cute::is_integral<TP>::value && cute::rank(TP{}) == 1 && cute::depth(TP{}) == 0,
      "Only integers allowed for TP at this time.");

  static_assert(cute::rank(ProcessorTiler_{}) == 4, "Expected rank-4 processor tiler.");
  static_assert(cute::rank(IterationTiler_{}) == 4, "Expected rank-4 iteration tiler.");

  static_assert(cute::rank(PeerDeviceMapping_{}) == 2,
      "PeerDeviceMapping must be rank-2 (device_idx, iter)");

  static_assert(cute::rank(IterationMappingM_{}) == 2,
      "IterationMappingM must be rank-2 (device_idx, iter).");
  static_assert(cute::rank(IterationMappingN_{}) == 2,
      "IterationMappingN must be rank-2 (device_idx, iter).");
  static_assert(cute::rank(IterationMappingK_{}) == 2,
      "IterationMappingK must be rank-2 (device_idx, iter).");
  static_assert(cute::rank(IterationMappingL_{}) == 2,
      "IterationMappingL must be rank-2 (device_idx, iter).");

  using ProcessorTiler = ProcessorTiler_;
  using IterationTiler = IterationTiler_;

  using PeerDeviceMapping = PeerDeviceMapping_;
  using IterationMappingM = IterationMappingM_;
  using IterationMappingN = IterationMappingN_;
  using IterationMappingK = IterationMappingK_;
  using IterationMappingL = IterationMappingL_;

  using ProcessorOffset = ProcessorOffset_;

  static constexpr bool KernelWritesArrivalFlag = KernelWritesArrivalFlag_;
  static constexpr bool MemcpyA = MemcpyA_;
  static constexpr bool MemcpyB = MemcpyB_;
  static constexpr bool HasMemcpy = MemcpyA || MemcpyB;

  static constexpr int NumBuffersA = NumBuffersA_;
  static constexpr int NumBuffersB = NumBuffersB_;
  static constexpr int NumBuffersC = NumBuffersC_;
  static constexpr int NumBuffersD = NumBuffersD_;

  static_assert(
      NumBuffersA > 0 ^
      NumBuffersB > 0 ^
      NumBuffersC > 0 ^
      NumBuffersD > 0,
      "Only one of the ABCD tensors can be buffered!");

  static constexpr bool BufferedOutput = NumBuffersC > 0 || NumBuffersD > 0;
  static constexpr bool RemoteC = NumBuffersC == 0 && NumBuffersD > 0;
  static constexpr bool RemoteD = NumBuffersD == 0 && NumBuffersC > 0;

  static_assert(not RemoteD, "Remote D is not supported yet.");

  // Host-side API: can_implement based on the GLOBAL problem shape
  template <typename ProblemShape>
  static bool
  can_implement_global(ProblemShape const& global_problem_shape) {
    auto [M, N, K, L] = append<4>(global_problem_shape, 1);

    auto [ptileM, ptileN, ptileK, ptileL] = ProcessorTiler{};
    auto [itileM, itileN, itileK, itileL] = IterationTiler{};

    auto tileM = ptileM * itileM;
    auto tileN = ptileN * itileN;
    auto tileK = ptileK * itileK;
    auto tileL = ptileL * itileL;

    return (M % tileM == 0) && (N % tileN == 0) && (K % tileK == 0) && (L % tileL == 0);
  }

  // Get the local (per-GPU) GEMM shape for a given global problem shape
  template <typename ProblemShape>
  static auto
  get_local_gemm_shape(ProblemShape const& global_problem_shape) {
    auto [M, N, K, L] = append<4>(global_problem_shape, 1);
    auto [ptileM, ptileN, ptileK, ptileL] = ProcessorTiler{};
    auto [itileM, itileN, itileK, itileL] = IterationTiler{};
    return cute::make_shape(M / ptileM / itileM, N / ptileN / itileN, K / ptileK / itileK);
  }

  // Get ABCD shapes for buffer sizing
  template <typename ProblemShape>
  static auto
  get_local_a_shape(ProblemShape const& problem_shape) {
    auto [M, N, K, L] = append<4>(problem_shape, 1);
    auto [itileM, itileN, itileK, itileL] = IterationTiler{};
    return cute::make_shape(M / itileM, K);
  }

  template <typename ProblemShape>
  static auto
  get_local_b_shape(ProblemShape const& problem_shape) {
    auto [M, N, K, L] = append<4>(problem_shape, 1);
    auto [itileM, itileN, itileK, itileL] = IterationTiler{};
    return cute::make_shape(N / itileN, K);
  }

  template <typename ProblemShape>
  static auto
  get_local_c_shape(ProblemShape const& problem_shape) {
    auto [M, N, K, L] = append<4>(problem_shape, 1);
    auto [itileM, itileN, itileK, itileL] = IterationTiler{};
    return cute::make_shape(M / itileM, N / itileN);
  }

  template <typename ProblemShape>
  static auto
  get_local_d_shape(ProblemShape const& problem_shape) {
    return get_local_c_shape(problem_shape);
  }

  // Iteration-to-peer mapping
  static int
  get_remote_peer_id(int device_idx, int iteration) {
    return (PeerDeviceMapping{}(device_idx, iteration) + int(ProcessorOffset{})) % int(TP{});
  }

  static auto
  get_peers_for_device(int device_idx) {
    int left_peer  = (device_idx - 1 + int(TP{})) % int(TP{});
    int right_peer = (device_idx + 1) % int(TP{});
    return cute::make_tuple(left_peer, right_peer);
  }

  // Construct the tensor slice for A/B/C/D at a given (device_idx, iteration)
  template <typename TensorA>
  static auto
  get_tensor_A(TensorA const& tensor_A, void* buffer, int device_idx, int iteration) {
    auto tile_idx = (IterationMappingM{}(device_idx, iteration) + int(ProcessorOffset{})) % int(TP{});
    auto [itileM, itileN, itileK, itileL] = IterationTiler{};
    if constexpr (NumBuffersA == 0) {
      return cute::local_tile(tensor_A, cute::make_shape(cute::shape<0>(tensor_A) / itileM, cute::shape<1>(tensor_A)), cute::make_coord(tile_idx, 0));
    } else {
      using ElementA = typename TensorA::value_type;
      auto buf_layout = cute::make_layout(cute::make_shape(cute::shape<0>(tensor_A) / itileM, cute::shape<1>(tensor_A)));
      return cute::make_tensor(reinterpret_cast<ElementA*>(buffer) + tile_idx * cute::cosize(buf_layout), buf_layout);
    }
  }

  template <typename TensorB>
  static auto
  get_tensor_B(TensorB const& tensor_B, void* buffer, int device_idx, int iteration) {
    auto tile_idx = (IterationMappingN{}(device_idx, iteration) + int(ProcessorOffset{})) % int(TP{});
    auto [itileM, itileN, itileK, itileL] = IterationTiler{};
    if constexpr (NumBuffersB == 0) {
      return cute::local_tile(tensor_B, cute::make_shape(cute::shape<0>(tensor_B) / itileN, cute::shape<1>(tensor_B)), cute::make_coord(tile_idx, 0));
    } else {
      using ElementB = typename TensorB::value_type;
      auto buf_layout = cute::make_layout(cute::make_shape(cute::shape<0>(tensor_B) / itileN, cute::shape<1>(tensor_B)));
      return cute::make_tensor(reinterpret_cast<ElementB*>(buffer) + tile_idx * cute::cosize(buf_layout), buf_layout);
    }
  }

  template <typename TensorC>
  static auto
  get_tensor_C(TensorC const& tensor_C, void* buffer, int device_idx, int iteration) {
    auto tile_idx = (IterationMappingM{}(device_idx, iteration) + int(ProcessorOffset{})) % int(TP{});
    auto [itileM, itileN, itileK, itileL] = IterationTiler{};
    if constexpr (NumBuffersC == 0 && NumBuffersD > 0) {
      // RemoteC: source is peer's D buffer
      using ElementC = typename TensorC::value_type;
      auto buf_layout = cute::make_layout(cute::make_shape(cute::shape<0>(tensor_C) / itileM, cute::shape<1>(tensor_C)));
      return cute::make_tensor(reinterpret_cast<ElementC*>(buffer) + tile_idx * cute::cosize(buf_layout), buf_layout);
    } else {
      return cute::local_tile(tensor_C, cute::make_shape(cute::shape<0>(tensor_C) / itileM, cute::shape<1>(tensor_C)), cute::make_coord(tile_idx, 0));
    }
  }

  template <typename TensorD>
  static auto
  get_tensor_D(TensorD const& tensor_D, void* buffer, int device_idx, int iteration) {
    auto tile_idx = (IterationMappingM{}(device_idx, iteration) + int(ProcessorOffset{})) % int(TP{});
    auto [itileM, itileN, itileK, itileL] = IterationTiler{};
    if constexpr (NumBuffersD == 0) {
      return cute::local_tile(tensor_D, cute::make_shape(cute::shape<0>(tensor_D) / itileM, cute::shape<1>(tensor_D)), cute::make_coord(tile_idx, 0));
    } else {
      using ElementD = typename TensorD::value_type;
      auto buf_layout = cute::make_layout(cute::make_shape(cute::shape<0>(tensor_D) / itileM, cute::shape<1>(tensor_D)));
      return cute::make_tensor(reinterpret_cast<ElementD*>(buffer) + tile_idx * cute::cosize(buf_layout), buf_layout);
    }
  }

};

} // namespace cutlass::distributed_xpu::schedules

///////////////////////////////////////////////////////////////////////////////
