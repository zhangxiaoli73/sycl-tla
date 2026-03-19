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
#include "allgather_gemm.hpp"

#ifndef SKIP_VERIFY
#define SKIP_VERIFY
#endif


// verification using cutlass::TensorRef
template <class ATensor, class BTensor, class CTensor>
bool gemm_verify_ref(sycl::queue& Q,
                     ATensor const& A, // (M,K)
                     BTensor const& B, // (N,K)
                     CTensor const& C) // (M,N)
{
  int M = size<0>(C);
  int N = size<1>(C);
  int K = size<1>(A);
  int L = 1;

  cutlass::TensorRef ref_A(&*A.data(),
                           cutlass::layout::RowMajor::packed({M, K}));
  cutlass::TensorRef ref_B(
      &*B.data(), cutlass::layout::RowMajor::packed({K, N})); // original

  using TC = typename CTensor::element_type;
  TC* ref_C_ptr = sycl::malloc_device<TC>(static_cast<std::size_t>(M) * N, Q);
  TC* ref_D_ptr = sycl::malloc_device<TC>(static_cast<std::size_t>(M) * N, Q);

  cutlass::TensorRef ref_C(ref_C_ptr,
                           cutlass::layout::RowMajor::packed({M, N}));
  cutlass::TensorRef ref_D(ref_D_ptr,
                           cutlass::layout::RowMajor::packed({M, N}));

  int npes = ishmem_n_pes();
  int mype = ishmem_my_pe();
  size_t local_size = (M * N) / npes;
  TC alpha = TC(npes);
  TC beta = TC(0);

  Q.fill(ref_C_ptr, TC(0), M * N).wait();
  Q.fill(ref_D_ptr, TC(0), M * N).wait();
  cutlass::reference::device::GemmComplex(
      {M, N, K}, alpha, ref_A, cutlass::ComplexTransform::kNone, ref_B,
      cutlass::ComplexTransform::kNone, beta, ref_C, ref_D);

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::device::BlockCompareEqual(
      &*ref_D.data() + mype * local_size, &*C.data() + mype * local_size,
      local_size);

  sycl::free(ref_D_ptr, Q);
  sycl::free(ref_C_ptr, Q);

  return passed;
}

template <class...>
class GemmVerifyKernelName;
template <class ATensor, class BTensor, class CTensor>
bool gemm_verify(sycl::queue& Q,
                 ATensor const& A, // (M,K)
                 BTensor const& B, // (N,K)
                 CTensor const& C) // (M,N)
{
  int m = size<0>(A);
  int n = size<0>(B);
  int k = size<1>(A);

  auto ok = sycl::malloc_shared<bool>(1, Q);
  *ok = true;

  // int npes = ishmem_n_pes();
  // int mype = ishmem_my_pe();
  // int nrows_pe = m / npes;
  int npes = 1;
  int mype = 0;
  int nrows_pe = 0;
  Q.parallel_for<GemmVerifyKernelName<ATensor, BTensor, CTensor>>(
       sycl::range<2>(m / npes, n),
       [=](sycl::item<2> id) {
         int i = id[0] + mype * nrows_pe, j = id[1];

         using AccType = typename CTensor::element_type;
         using SignedAccType = ensure_signed_t<AccType>;

         auto c = AccType(0);
         for (int h = 0; h < k; h++)
           c += AccType(A(i, h)) * AccType(B(j, h));

         c = c * npes;
         auto tol = AccType(1e-5f * k);
         if (std::abs(SignedAccType(c - AccType(C(i, j)))) > tol) {
// #ifndef SHOW_DIFF
//            printf("Error at (%d,%d): got %f, expected %f\n", i, j,
//                   double(C(i, j)), double(c));
// #endif
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

  if (ishmem_my_pe() == 0)
    std::cout << type_str<TA>() << " (" << LayoutA << ") x " << type_str<TB>()
              << " (" << LayoutB << ") -> " << type_str<TC>() << ": \n";

  // Transpose B to match CuTe conventions
  constexpr char tlayoutB = LayoutB ^ ('R' ^ 'C');

  // Prepare data:
  auto A = make_shared_usm_tensor<TA, LayoutA>(Q, m, k);
  auto B = make_shared_usm_tensor<TB, tlayoutB>(Q, n, k);

  auto C = make_shared_usm_tensor<TC, 'R'>(Q, world_size, m, n);

  std::cout << "usm allocated for A, B, C\n";

  // random_fill(A);
  // random_fill(B);
  uint64_t seed = 11;
  cutlass::initialize_block(&*A.data(), m * k, seed + 2023);
  cutlass::initialize_block(&*B.data(), n * k, seed + 2022);

  Q.fill(&*C.data(), TC(0), world_size * m * n).wait();

  std::cout << "usm filled for A, B, C\n";

#ifndef SKIP_VERIFY
  auto A_ref = make_shared_usm_tensor<float, LayoutA>(Q, m, k);
  auto B_ref = make_shared_usm_tensor<float, tlayoutB>(Q, n, k);

  copy(A, A_ref);
  copy(B, B_ref);
#endif

  subbyte_pack(A);
  subbyte_pack(B);

  ishmem_barrier_all();

  std::vector<double> durations;

  // std::cout << "after ishmem_barrier_all\n";
  AllGatherGemm<TA, TB, TC, LayoutA, tlayoutB> ag_gemm(m, n, k, rank, world_size, Q);
  for (int i = 0; i < 30; i++) {
    Q.wait();
    std::cout << "node: " << rank << ", running iteration " << i << "======================" <<std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "node: " << rank << ", ag_gemm running\n";
    ag_gemm.run(A, B, C, Q);
    std::cout << "node: " << rank << ", ag_gemm run done\n";
    auto stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration_ms = stop - start;
    // if (i >= 20) {
    durations.push_back(duration_ms.count());
    // ishmem_barrier_all();
    // }
  }

  for (int i=0; i < 30; i++) {
    std::cout << durations[i] << " ";
  }
  std::cout << std::endl;
  

  std::ostringstream message;
  message << ishmem_my_pe() << " ";
#ifdef SKIP_VERIFY
  const bool ok = true;
  message << "verification skipped";
#else

  // bool ok = gemm_verify_ref(Q, A_ref, B_ref, C);
  auto local_c = C(rank, _, _);
  bool ok = gemm_verify(Q, A_ref, B_ref, local_c);

  if (ok) {
    // verify remote part
    for (int i = rank + 1; i < (world_size + rank); i++) {
      int target_rank = (i + 1) % world_size;
      auto inputA_buffer = ag_gemm.get_local_input_buffer();
      auto mk_layout = make_layout(make_shape(m, k), LayoutRight{});
      auto remote_A = make_tensor(make_gmem_ptr(static_cast<TA*>(inputA_buffer.data().get()) + target_rank * m * k), mk_layout);
      std::cout << "copying A" << std::endl;

      // Q.memcpy(A_ref.data().get(), remote_A.data().get(), m * k * sizeof(TA)).wait();
      // failed to copy ishmem allocated mem
      // copy(remote_A, A_ref);

      std::cout << "copying A done" << std::endl;
      auto local_c_2 = C(target_rank, _, _);
      std::cout << "gemm verifying A" << std::endl;
      ok = gemm_verify(Q, remote_A, B_ref, local_c_2);
    }
    

    // failed to run due to ishmem allocated mem copying
    // ok = gemm_verify(Q, A_ref, B_ref, local_c_2);

    std::cout << "gemm verifying A done" << std::endl;
  }

  message << (ok ? "passed" : "failed");
#endif

  // if (true) {
  //   // Test performance:
  //   const int timing_iterations = 100;
  //   GPU_Clock timer;

  //   timer.start();
  //   for (int i = 0; i < timing_iterations; ++i)
  //     gemm_cute<decltype(A), decltype(B), decltype(C), TA, TB, layoutA,
  //               layoutB>(Q, A, B, C);
  //   ishmemx_barrier_all_on_queue(Q).wait();

  //   double avg = timer.seconds() / timing_iterations;
  //   double tops = (2.0 * m * n * k) * 1e-12;

  //   char buf[100];
  //   std::sprintf(buf, ", %4.3f TF/s\n", tops / avg);
  //   message << buf;
  //   // printf(", %4.3f TF/s", tops / avg, avg * 1000);
  // }
  std::cout << message.str();

  free_usm(Q, A.data().get());
  free_usm(Q, B.data().get());
  free_usm(Q, C.data().get());

#ifndef SKIP_VERIFY
  free_usm(Q, A_ref.data().get());
  free_usm(Q, B_ref.data().get());
#endif

  // std::cout << '\n';

  // Pause for a short period of time to allow the GPU to cool.
  static bool first = true;
  if (first)
    first = false;
  else
    sleep(1);
}



int main(int argc, char** argv) {
  auto shift = [&] { return (argc-- > 0) ? *argv++ : nullptr; };

  auto parse_size = [&] {
    static constexpr int default_size = 1024;
    if (auto e = shift())
      return atoi(e);
    else
      return default_size;
  };

  (void) shift();

  // std::cout << "Running allgather_gemm_main\n";

  // intra-node for now
  // int local_world_size = world_size;

  //================bfloat16_t, bfloat16_t, float, 'R', 'R' for now=============================
  
  ishmem_init();
  std::cout << "ishmem_init done\n";
  {
    int world_size = ishmem_n_pes();
    int rank = ishmem_my_pe();
    // int local_world_size = ishmem_local_n_pes();
    int local_world_size = world_size;

    std::cout << "got rank and world_size, " << rank << ":" << world_size << "\n";

    auto m = parse_size();
    auto n = parse_size();
    auto k = parse_size();

    if(m < world_size * 256) {
      if(ishmem_my_pe() == 0)
        std::cout << m <<  " is too small for the number of PEs, increase M size or reduce number of PEs\n";
      ishmem_finalize();
      return 0;
    }

    int max_m_dim = m * world_size;
    int k_dim = k;

    using input_dtype = cute::bfloat16_t;

    sycl::queue Q({sycl::property::queue::in_order()});

    test_case<input_dtype, input_dtype, float, 'R', 'R'>(Q, m, n, k, rank, world_size);
  }
  ishmem_finalize();
}
