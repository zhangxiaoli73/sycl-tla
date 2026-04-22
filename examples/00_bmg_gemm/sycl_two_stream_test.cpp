/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kChunkElements = 1 << 18;
constexpr std::size_t kNumChunks = 4;
constexpr std::size_t kTotalElements = kChunkElements * kNumChunks;
constexpr int kComputeIters = 64;

float reference_value(float x) {
  float value = x;
  for (int iter = 0; iter < kComputeIters; ++iter) {
    value = value * 1.0001f + 0.25f;
  }
  return value;
}

}  // namespace

int main() {
  auto async_handler = [](sycl::exception_list exceptions) {
    for (auto const& exception : exceptions) {
      try {
        std::rethrow_exception(exception);
      } catch (sycl::exception const& ex) {
        std::cerr << "Asynchronous SYCL exception: " << ex.what() << std::endl;
      }
    }
  };

  try {
    sycl::device device{sycl::gpu_selector_v};
    sycl::context context{device, async_handler};

    sycl::queue stream0{context, device, async_handler, sycl::property::queue::in_order{}};
    sycl::queue stream1{context, device, async_handler, sycl::property::queue::in_order{}};

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Submitting compute kernels on stream0 and copy kernels on stream1." << std::endl;

    std::vector<float> host_input(kTotalElements);
    std::vector<float> host_output(kTotalElements, 0.0f);

    float* input = sycl::malloc_device<float>(kTotalElements, stream0);
    float* staging = sycl::malloc_device<float>(kTotalElements, stream0);
    float* output = sycl::malloc_device<float>(kTotalElements, stream0);

    if (!input || !staging || !output) {
      std::cerr << "USM allocation failed." << std::endl;
      if (input) {
        sycl::free(input, stream0);
      }
      if (staging) {
        sycl::free(staging, stream0);
      }
      if (output) {
        sycl::free(output, stream0);
      }
      return 1;
    }

    for (std::size_t idx = 0; idx < kTotalElements; ++idx) {
      host_input[idx] = static_cast<float>(idx % 251) * 0.5f;
    }

    stream0.memcpy(input, host_input.data(), kTotalElements * sizeof(float)).wait();

    auto submit_begin = std::chrono::steady_clock::now();
    auto execute_begin = submit_begin;

    for (std::size_t chunk = 0; chunk < kNumChunks; ++chunk) {
      std::size_t begin = chunk * kChunkElements;

      stream0.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(kChunkElements), [=](sycl::id<1> idx) {
          std::size_t offset = begin + idx[0];
          float value = input[offset];
          for (int iter = 0; iter < kComputeIters; ++iter) {
            value = value * 1.0001f + 0.25f;
          }
          staging[offset] = value;
        });
      });

      stream1.memcpy(output + begin, staging + begin, kChunkElements * sizeof(float));

      if ((chunk + 1) % 2 == 0) {
        stream0.wait();
        stream1.wait();
      }
    }

    auto submit_end = std::chrono::steady_clock::now();

    stream0.wait();
    stream1.wait();
    auto execute_end = std::chrono::steady_clock::now();

    stream1.memcpy(host_output.data(), output, kTotalElements * sizeof(float)).wait();

    auto submit_us = std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_begin).count();
    auto execute_us = std::chrono::duration_cast<std::chrono::microseconds>(execute_end - execute_begin).count();

    std::cout << "Chunk loop submit time: " << submit_us << " us" << std::endl;
    std::cout << "Chunk loop end-to-end time (with queue waits): " << execute_us << " us" << std::endl;

    std::size_t mismatches = 0;
    for (std::size_t idx = 0; idx < kTotalElements; ++idx) {
      float expected = reference_value(host_input[idx]);
      float diff = std::fabs(host_output[idx] - expected);
      if (diff > 1e-4f) {
        if (mismatches < 8) {
          std::cerr << "Mismatch at " << idx << ": got " << host_output[idx]
                    << ", expected " << expected << std::endl;
        }
        ++mismatches;
      }
    }

    sycl::free(input, stream0);
    sycl::free(staging, stream0);
    sycl::free(output, stream0);

    if (mismatches != 0) {
      std::cerr << "Verification failed with " << mismatches << " mismatches." << std::endl;
      return 1;
    }

    std::cout << "PASS: compute kernels were submitted on stream0, copy kernels were submitted on stream1, and results verified." << std::endl;
    return 0;
  } catch (sycl::exception const& ex) {
    std::cerr << "SYCL exception: " << ex.what() << std::endl;
  } catch (std::exception const& ex) {
    std::cerr << "Exception: " << ex.what() << std::endl;
  }

  return 1;
}