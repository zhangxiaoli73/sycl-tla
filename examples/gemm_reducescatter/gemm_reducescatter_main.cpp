/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <chrono>
#include <mpi.h>
#include "gemm_reducescatter.hpp"

#ifndef SKIP_VERIFY
#define SKIP_VERIFY
#endif


template <class...>
class GemmVerifyKernelName;
template <class ATensor, class BTensor, class CTensor>
bool gemm_verify(sycl::queue& Q,
                 ATensor const& A, // (M,K)
                 BTensor const& B, // (N,K)
                 CTensor const& C, // (M,N)
                 int world_size)
{
  int m = size<0>(A);
  int n = size<0>(B);
  int k = size<1>(A);

  auto ok = sycl::malloc_shared<bool>(1, Q);
  *ok = true;

  Q.parallel_for<GemmVerifyKernelName<ATensor, BTensor, CTensor>>(
       sycl::range<2>(m, n),
       [=](sycl::item<2> id) {
         int i = id[0], j = id[1];

         using AccType = typename CTensor::element_type;
         using SignedAccType = ensure_signed_t<AccType>;

         auto c = AccType(0);
         for (int h = 0; h < k; h++)
           c += AccType(A(i, h)) * AccType(B(j, h));

         // After allreduce, C = world_size * (A * B)
         c = c * world_size;
         auto tol = AccType(1e-5f * k);
         if (std::abs(SignedAccType(c - AccType(C(i, j)))) > tol) {
           *ok = false;
         }
       })
      .wait();

  bool read_ok = *ok;
  sycl::free(ok, Q);
  return read_ok;
}

template <typename TA, typename TB, typename TC, char LayoutA = 'R',
          char LayoutB = 'R'>
void test_case(sycl::queue& Q, int m, int n, int k, int rank, int world_size) {

  if (rank == 0 && rs_log_enabled())
    std::cout << type_str<TA>() << " (" << LayoutA << ") x " << type_str<TB>()
              << " (" << LayoutB << ") -> " << type_str<TC>() << ": \n";

  // Transpose B to match CuTe conventions
  constexpr char tlayoutB = LayoutB ^ ('R' ^ 'C');

  // Prepare data:
  auto A = make_shared_usm_tensor<TA, LayoutA>(Q, m, k);
  auto B = make_shared_usm_tensor<TB, tlayoutB>(Q, n, k);
  auto C = make_shared_usm_tensor<TC, 'R'>(Q, m, n);
  auto D = make_shared_usm_tensor<TC, 'R'>(Q, m, n / world_size); // reduce-scatter output

  if (rs_log_enabled()) {
    std::cout << "node: " << rank << ", usm allocated for A, B, C\n";
  }

  uint64_t seed = 11;
  cutlass::initialize_block(&*A.data(), m * k, seed + 2023);
  cutlass::initialize_block(&*B.data(), n * k, seed + 2022);
  Q.fill(&*C.data(), TC(0), m * n).wait();
  Q.fill(&*D.data(), TC(0), m * (n / world_size)).wait();

  if (rs_log_enabled()) {
    std::cout << "node: " << rank << ", usm filled for A, B, C, D\n";
  }

#ifndef SKIP_VERIFY
  auto A_ref = make_shared_usm_tensor<float, LayoutA>(Q, m, k);
  auto B_ref = make_shared_usm_tensor<float, tlayoutB>(Q, n, k);
  copy(A, A_ref);
  copy(B, B_ref);
#endif

  subbyte_pack(A);
  subbyte_pack(B);

  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int num_iters = 20;
  constexpr int warmup_iters = 10;

  // --- Run fused path ---
  std::vector<double> fused_durations;
  {
    // Force fusion enabled
    GemmAllReduce<TA, TB, TC, LayoutA, tlayoutB> gemm_ar_fused(
        m, n, k, rank, world_size, A, B, C, D, Q, true);

    for (int i = 0; i < num_iters; i++) {
      if (rs_log_enabled()) {
        std::cout << "[debug] rank " << rank << " fused iter " << i << " begin\n";
      }
      Q.wait();
      Q.fill(&*C.data(), TC(0), m * n).wait();
      Q.fill(&*D.data(), TC(0), m * (n / world_size)).wait();
      MPI_Barrier(MPI_COMM_WORLD);

      auto start = std::chrono::high_resolution_clock::now();
      gemm_ar_fused.run_fused(A, B, C, D, Q);
      Q.wait();
      auto stop = std::chrono::high_resolution_clock::now();
      if (i >= warmup_iters) {
        const std::chrono::duration<double, std::milli> duration_ms = stop - start;
        fused_durations.push_back(duration_ms.count());
      }
      if (rs_log_enabled()) {
        std::cout << "[debug] rank " << rank << " fused iter " << i << " end\n";
      }
    }
  }

  // --- Run separate (non-fused) path ---
  std::vector<double> separate_durations;
  {
    GemmAllReduce<TA, TB, TC, LayoutA, tlayoutB> gemm_ar_sep(
        m, n, k, rank, world_size, A, B, C, D, Q, false);

    for (int i = 0; i < num_iters; i++) {
      if (rs_log_enabled()) {
        std::cout << "[debug] rank " << rank << " separate iter " << i << " begin\n";
      }
      Q.wait();
      Q.fill(&*C.data(), TC(0), m * n).wait();
      MPI_Barrier(MPI_COMM_WORLD);

      auto start = std::chrono::high_resolution_clock::now();
      gemm_ar_sep.run_gemm(A, B, C, Q);
      Q.wait();
      auto stop = std::chrono::high_resolution_clock::now();
      if (i >= warmup_iters) {
        const std::chrono::duration<double, std::milli> duration_ms = stop - start;
        separate_durations.push_back(duration_ms.count());
      }
      if (rs_log_enabled()) {
        std::cout << "[debug] rank " << rank << " separate iter " << i << " end\n";
      }
    }
  }

  // --- Print comparison ---
  if (rank == 0) {
    auto avg = [&](const std::vector<double>& v) {
      double sum = 0;
      for (double duration : v) sum += duration;
      return sum / v.size();
    };

    double fused_avg = avg(fused_durations);
    double separate_avg = avg(separate_durations);
    double speedup = separate_avg / fused_avg;

    std::cout << "\n=== GEMM + AllReduce Benchmark (M=" << m << ", N=" << n << ", K=" << k
              << ", ranks=" << world_size << ") ===\n";
    std::cout << "Fused timings (ms, warmup excluded):    ";
    for (auto d : fused_durations) std::cout << d << " ";
    std::cout << "\nSeparate timings (ms, warmup excluded): ";
    for (auto d : separate_durations) std::cout << d << " ";
    std::cout << "\n\nFused avg    (excl warmup): " << fused_avg << " ms\n";
    std::cout << "Separate avg (excl warmup): " << separate_avg << " ms\n";
    std::cout << "Speedup (separate/fused):   " << speedup << "x\n";
    std::cout << "==========================================\n";
  }

  std::ostringstream message;
  message << "rank " << rank << ": ";
#ifdef SKIP_VERIFY
  message << "verification skipped";
#else
  bool ok = gemm_verify(Q, A_ref, B_ref, C, world_size);
  message << (ok ? "passed" : "failed");
#endif

  std::cout << message.str() << std::endl;

  sycl::free(A.data().get(), Q);
  sycl::free(B.data().get(), Q);
  sycl::free(C.data().get(), Q);
  sycl::free(D.data().get(), Q);

#ifndef SKIP_VERIFY
  sycl::free(A_ref.data().get(), Q);
  sycl::free(B_ref.data().get(), Q);
#endif
}



int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rs_log_enabled()) {
    std::cout << "MPI initialized, rank " << rank << " / " << world_size << "\n";
  }

  auto shift = [&] { return (argc-- > 0) ? *argv++ : nullptr; };

  auto parse_size = [&](int default_val) {
    if (auto e = shift())
      return atoi(e);
    else
      return default_val;
  };

  (void) shift();

  auto m = parse_size(8192);
  auto n = parse_size(4096);
  auto k = parse_size(1024);

  if (m < 256) {
    if (rank == 0)
      std::cout << m << " is too small, M must be >= 256\n";
    MPI_Finalize();
    return 0;
  }

  using input_dtype = cute::bfloat16_t;

  sycl::async_handler async_handler = [rank](sycl::exception_list exceptions) {
    for (auto const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (const sycl::exception& ex) {
        std::cerr << "[rank " << rank << "] [async sycl exception] " << ex.what() << "\n";
      }
    }
  };

  // Explicitly select device based on rank
  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (rs_log_enabled()) {
    std::cout << "[rank " << rank << "] Found " << devices.size() << " GPU device(s):\n";
    for (size_t i = 0; i < devices.size(); ++i) {
      std::cout << "  [" << i << "] " << devices[i].get_info<sycl::info::device::name>() << "\n";
    }
  }
  if (devices.empty()) {
    std::cerr << "No GPU devices found\n";
    MPI_Finalize();
    return 1;
  }
  auto device = devices[rank % devices.size()];
  if (rs_log_enabled()) {
    std::cout << "[rank " << rank << "] Selected device[" << (rank % devices.size()) << "]: " << device.get_info<sycl::info::device::name>() << "\n";
  }

  sycl::queue Q(device, async_handler,
                {sycl::property::queue::in_order()});

  try {
    if (rs_log_enabled()) {
      std::cout << "[debug] rank " << rank << " entering test_case\n";
    }
    test_case<input_dtype, input_dtype, float, 'R', 'R'>(Q, m, n, k, rank, world_size);
    if (rs_log_enabled()) {
      std::cout << "[debug] rank " << rank << " test_case completed\n";
    }
  } catch (const sycl::exception& ex) {
    std::cerr << "[rank " << rank << "] [sync sycl exception] " << ex.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  } catch (const std::exception& ex) {
    std::cerr << "[rank " << rank << "] [std exception] " << ex.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
