/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kChunkElements = 8192 * 1024;  // kTotalElements = 8192 * 4096
constexpr std::size_t kNumChunks = 10;
constexpr std::size_t kTotalElements = kChunkElements * kNumChunks;
constexpr int kComputeIters = 128;
constexpr std::size_t kDefaultLoopCount = 20;

float reference_value(float x) {
  float value = x;
  for (int iter = 0; iter < kComputeIters; ++iter) {
    value = value * 1.0001f + 0.25f;
  }
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t loop_count = kDefaultLoopCount;
  if (argc >= 2) {
    std::string arg = argv[1];
    if (arg.rfind("--loop-count=", 0) == 0) {
      loop_count = static_cast<std::size_t>(std::strtoull(arg.c_str() + 13, nullptr, 10));
    } else {
      loop_count = static_cast<std::size_t>(std::strtoull(arg.c_str(), nullptr, 10));
    }
  }

  if (loop_count == 0) {
    std::cerr << "loop_count must be > 0" << std::endl;
    return 1;
  }

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
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpu_devices.size() < 2) {
      std::cerr << "This test requires at least 2 GPU devices: device0 and device1." << std::endl;
      return 1;
    }

    sycl::device device0 = gpu_devices[0];
    sycl::device device1 = gpu_devices[1];
    sycl::context context{std::vector<sycl::device>{device0, device1}, async_handler};

    sycl::queue stream0{context, device0, async_handler,
          sycl::property_list{sycl::property::queue::in_order{},
              sycl::property::queue::enable_profiling{}}};
    sycl::queue stream1{context, device0, async_handler,
          sycl::property_list{sycl::property::queue::in_order{},
              sycl::property::queue::enable_profiling{}}};
    sycl::queue stream_device1{context, device1, async_handler,
          sycl::property_list{sycl::property::queue::in_order{},
              sycl::property::queue::enable_profiling{}}};

    std::cout << "Running on device0: "
              << device0.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Running on device1: "
              << device1.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "stream0/stream1 on device0, stream_device1 on device1." << std::endl;
    std::cout << "Submitting compute kernels on stream0 and copy operations on stream1 (device0)." << std::endl;
    std::cout << "Loop count: " << loop_count << ", chunks per loop: " << kNumChunks << std::endl;

    float* input = sycl::malloc_device<float>(kTotalElements, stream0);
    float* staging = sycl::malloc_device<float>(kTotalElements, stream0);
    float* dst_buffer = sycl::malloc_device<float>(kTotalElements, stream1);
    // src_buffer on device1 (allocated via stream_device1)
    float* src_buffer = sycl::malloc_device<float>(kTotalElements, stream_device1);

    if (!input || !staging || !dst_buffer || !src_buffer) {
      std::cerr << "USM allocation failed." << std::endl;
      if (input) {
        sycl::free(input, stream0);
      }
      if (staging) {
        sycl::free(staging, stream0);
      }
      if (dst_buffer) {
        sycl::free(dst_buffer, stream1);
      }
      if (src_buffer) {
        sycl::free(src_buffer, stream_device1);
      }
      return 1;
    }
    //initialize input and src_buffer with the same data
    std::vector<float> host_input(kTotalElements);
    std::vector<float> host_output(kTotalElements, 0.0f);
    for (std::size_t i = 0; i < kTotalElements; ++i) {
      host_input[i] = static_cast<float>(i) * 0.001f;
    }
    stream0.memcpy(input, host_input.data(), kTotalElements * sizeof(float)).wait();
    stream_device1.memcpy(src_buffer, host_input.data(), kTotalElements * sizeof(float)).wait();

    std::vector<sycl::event> loop_start_events;
    std::vector<sycl::event> loop_end_events;
    std::vector<double> measured_loop_us;
    if (loop_count > 11) {
      std::size_t measured_count = loop_count - 11;
      loop_start_events.reserve(measured_count);
      loop_end_events.reserve(measured_count);
      measured_loop_us.reserve(measured_count);
    }

    for (std::size_t loop_idx = 0; loop_idx < loop_count; ++loop_idx) {
      bool measure_this_loop = (loop_idx > 10);
      sycl::event loop_start_event;
      sycl::event loop_end_event;

      if (measure_this_loop) {
        // Start marker synchronized with both device0 queues.
        loop_start_event = stream0.ext_oneapi_submit_barrier({stream1.ext_oneapi_submit_barrier()});
      }

      for (std::size_t chunk = 0; chunk < kNumChunks; ++chunk) {
        std::size_t begin = chunk * kChunkElements;

        stream0.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::range<1>(kChunkElements), [=](sycl::id<1> idx) {
            std::size_t offset = begin + idx[0];
            float value = input[offset];
            for (int iter = 0; iter < kComputeIters; ++iter) {
              value = value * 1.0001f + loop_idx * 0.01f + 0.25f;
            }
            staging[offset] = value;
          });
        });

        // Direct cross-device copy in shared context: device1(src_buffer) -> device0(dst_buffer).
        stream1.memcpy(dst_buffer + begin, src_buffer + begin, kChunkElements * 2);
      }

      // End marker synchronized with both device0 queues.
      loop_end_event = stream0.ext_oneapi_submit_barrier({stream1.ext_oneapi_submit_barrier()});
      stream0.wait();

      if (measure_this_loop) {
        loop_start_events.push_back(loop_start_event);
        loop_end_events.push_back(loop_end_event);
      }
    }

    stream0.wait();
    stream1.wait();

    for (std::size_t i = 0; i < loop_start_events.size(); ++i) {
      uint64_t start_ns = loop_start_events[i].get_profiling_info<sycl::info::event_profiling::command_end>();
      uint64_t end_ns = loop_end_events[i].get_profiling_info<sycl::info::event_profiling::command_end>();
      measured_loop_us.push_back(static_cast<double>(end_ns - start_ns) / 1000.0);
    }

    stream0.memcpy(host_output.data(), dst_buffer, kTotalElements * sizeof(float)).wait();

    if (!measured_loop_us.empty()) {
      std::cout << "Chunk-loop time samples (SYCL event, loop_idx > 10), unit: us" << std::endl;
      for (std::size_t i = 0; i < measured_loop_us.size(); ++i) {
        std::cout << "sample[" << i << "] duration_us=" << measured_loop_us[i] << std::endl;
      }
    } else {
      std::cout << "No measured loops: require loop_count > 11 to collect stats." << std::endl;
    }

    sycl::free(input, stream0);
    sycl::free(staging, stream0);
    sycl::free(dst_buffer, stream1);
    sycl::free(src_buffer, stream1);

    std::cout << "PASS: compute kernels were submitted on stream0 and copy operations were submitted on stream1." << std::endl;
    return 0;
  } catch (sycl::exception const& ex) {
    std::cerr << "SYCL exception: " << ex.what() << std::endl;
  } catch (std::exception const& ex) {
    std::cerr << "Exception: " << ex.what() << std::endl;
  }

  return 1;
}