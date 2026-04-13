/*
 * Device-to-Device (D2D) Round-Trip Latency Benchmark
 * ===================================================
 *
 * This benchmark keeps the microarchitecture access path aligned with
 * pingpong_latency.cpp:
 *   - polling with lsc_load.ugm.uc.uc
 *   - reply writes with lsc_atomic_store.ugm
 *   - ordering with lsc_fence.ugm.invalidate/evict
 *
 * Difference from the PCIe version:
 *   - no host_buf
 *   - one extra device control buffer for start/ready/done handshake
 */

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>

static constexpr uint32_t NUM_ROUNDS = 10000u;
static constexpr uint32_t NUM_WARMUP = 200u;

struct alignas(64) DeviceBuf {
	uint32_t ctr;
	uint8_t pad[60];
};

struct alignas(64) DeviceCtrlBuf {
	uint32_t start;
	uint32_t ready_a;
	uint32_t ready_b;
	uint32_t done_a;
	uint32_t done_b;
	uint8_t pad[44];
};

static_assert(sizeof(DeviceBuf) == 64, "DeviceBuf must be 64 bytes");
static_assert(sizeof(DeviceCtrlBuf) == 64, "DeviceCtrlBuf must be 64 bytes");

static uint32_t poll_device_u32(sycl::queue &q,
								const uint32_t *device_addr,
								uint32_t expected,
								const char *tag = "") {
	using clk = std::chrono::steady_clock;
	auto deadline = clk::now() + std::chrono::seconds(2);
	bool warned = false;
	for (;;) {
		(void)q;
		uint32_t v;
		__asm__ volatile ("clflushopt (%0)" :: "r"(device_addr) : "memory");
		__asm__ volatile ("movl (%1), %0" : "=r"(v) : "r"(device_addr) : "memory");
		if (v == expected) return v;
		__asm__ volatile ("pause" ::: "memory");
		if (!warned && clk::now() > deadline) {
			std::cerr << "[poll timeout] " << tag << ": expected="
							<< expected << " got=" << v << "\n";
			warned = true;
			deadline += std::chrono::seconds(10);
		}
	}
}

static double get_gpu_tick_ns(ze_device_handle_t ze_dev) {
	ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
	ze_result_t rc = zeDeviceGetProperties(ze_dev, &props);
	if (rc != ZE_RESULT_SUCCESS) {
		throw std::runtime_error("zeDeviceGetProperties failed");
	}
	return static_cast<double>(props.timerResolution);
}

static sycl::event submit_pingpong_kernel(sycl::queue &kq,
										  DeviceBuf *send_buf,
										  DeviceBuf *recv_buf,
										  DeviceCtrlBuf *ctrl,
										  uint32_t num_rounds,
										  bool is_initiator) {
	return kq.submit([&](sycl::handler &h) {
		h.parallel_for(
			sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(16)),
			[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
				if (item.get_local_id(0) != 0) return;

				uint32_t *sp = &send_buf->ctr;
				uint32_t *rp = &recv_buf->ctr;
				uint32_t *startp = &ctrl->start;
				uint32_t *readyp = is_initiator ? &ctrl->ready_a : &ctrl->ready_b;
				uint32_t *donep = is_initiator ? &ctrl->done_a : &ctrl->done_b;

				{
					uint32_t one = 1u;
#ifdef __SYCL_DEVICE_ONLY__
					__asm__ volatile (
						"lsc_atomic_store.ugm (M1, 16)"
						"  %%null:d32 flat[%0]:a64 %1 %%null\n"
						: : "rw"(readyp), "rw"(one)
					);
					__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
					*readyp = one;
#endif
				}

					if (is_initiator) {
						uint32_t ready_b;
						do {
#ifdef __SYCL_DEVICE_ONLY__
							__asm__ volatile (
								"lsc_load.ugm.uc.uc (M1, 16) %0:d32 flat[%1]:a64\n"
								: "=rw"(ready_b) : "rw"(&ctrl->ready_b)
							);
#else
							ready_b = ctrl->ready_b;
#endif
						} while (ready_b != 1u);

						{
							uint32_t one = 1u;
#ifdef __SYCL_DEVICE_ONLY__
							__asm__ volatile (
								"lsc_atomic_store.ugm (M1, 16)"
								"  %%null:d32 flat[%0]:a64 %1 %%null\n"
								: : "rw"(startp), "rw"(one)
							);
							__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
							*startp = one;
#endif
						}
					} else {
						uint32_t started;
						do {
#ifdef __SYCL_DEVICE_ONLY__
							__asm__ volatile (
								"lsc_load.ugm.uc.uc (M1, 16) %0:d32 flat[%1]:a64\n"
								: "=rw"(started) : "rw"(startp)
							);
#else
							started = *startp;
#endif
						} while (started != 1u);
					}

				for (uint32_t r = 0; r < num_rounds; r++) {
#ifdef __SYCL_DEVICE_ONLY__
					__asm__ volatile ("lsc_fence.ugm.invalidate.tile\n" : : :);
#endif

					const uint32_t req = 2u * r + 1u;
					const uint32_t rep = req + 1u;
					uint32_t val;

					if (is_initiator) {
#ifdef __SYCL_DEVICE_ONLY__
						__asm__ volatile (
							"lsc_atomic_store.ugm (M1, 16)"
							"  %%null:d32 flat[%0]:a64 %1 %%null\n"
							: : "rw"(sp), "rw"(req)
						);
						__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
						*sp = req;
#endif

						do {
#ifdef __SYCL_DEVICE_ONLY__
							__asm__ volatile (
								"lsc_load.ugm.uc.uc (M1, 16) %0:d32 flat[%1]:a64\n"
								: "=rw"(val) : "rw"(rp)
							);
#else
							val = *rp;
#endif
						} while (val != rep);
					} else {
						do {
#ifdef __SYCL_DEVICE_ONLY__
							__asm__ volatile (
								"lsc_load.ugm.uc.uc (M1, 16) %0:d32 flat[%1]:a64\n"
								: "=rw"(val) : "rw"(sp)
							);
#else
							val = *sp;
#endif
						} while (val != req);

#ifdef __SYCL_DEVICE_ONLY__
						__asm__ volatile ("lsc_fence.ugm.invalidate.tile\n" : : :);
						__asm__ volatile (
							"lsc_atomic_store.ugm (M1, 16)"
							"  %%null:d32 flat[%0]:a64 %1 %%null\n"
							: : "rw"(rp), "rw"(rep)
						);
						__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
						*rp = rep;
#endif
					}

					if (is_initiator) {
						uint32_t step = r + 1u;
#ifdef __SYCL_DEVICE_ONLY__
						__asm__ volatile (
							"lsc_atomic_store.ugm (M1, 16)"
							"  %%null:d32 flat[%0]:a64 %1 %%null\n"
							: : "rw"(donep), "rw"(step)
						);
						__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
						*donep = step;
#endif
					}
				}

				if (!is_initiator) {
					uint32_t one = 1u;
#ifdef __SYCL_DEVICE_ONLY__
					__asm__ volatile (
						"lsc_atomic_store.ugm (M1, 16)"
						"  %%null:d32 flat[%0]:a64 %1 %%null\n"
						: : "rw"(donep), "rw"(one)
					);
					__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
					*donep = one;
#endif
				}
			}
		);
	});
}

static double run_session(sycl::context &ctx,
						  sycl::device &dev,
						  uint32_t rounds,
						  double gpu_tick_ns,
						  ze_device_handle_t ze_dev) {
	sycl::queue q_a(ctx, dev, sycl::property::queue::in_order{});
	sycl::queue q_b(ctx, dev, sycl::property::queue::in_order{});
	sycl::queue q_ctrl(ctx, dev, sycl::property::queue::in_order{});

	auto *dev_a = sycl::malloc_device<DeviceBuf>(1, q_a);
	auto *dev_b = sycl::malloc_device<DeviceBuf>(1, q_b);
	auto *ctrl = sycl::malloc_device<DeviceCtrlBuf>(1, q_ctrl);

	if (!dev_a || !dev_b || !ctrl) {
		if (dev_a) sycl::free(dev_a, q_a);
		if (dev_b) sycl::free(dev_b, q_b);
		if (ctrl) sycl::free(ctrl, q_ctrl);
		throw std::runtime_error("USM allocation failed");
	}

	q_a.memset(dev_a, 0, sizeof(DeviceBuf)).wait();
	q_b.memset(dev_b, 0, sizeof(DeviceBuf)).wait();
	q_ctrl.memset(ctrl, 0, sizeof(DeviceCtrlBuf)).wait();

	auto evt_a = submit_pingpong_kernel(q_a, dev_a, dev_b, ctrl, rounds, true);
	auto evt_b = submit_pingpong_kernel(q_b, dev_a, dev_b, ctrl, rounds, false);

	poll_device_u32(q_ctrl, &ctrl->ready_a, 1u, "ready_a");
	poll_device_u32(q_ctrl, &ctrl->ready_b, 1u, "ready_b");
	poll_device_u32(q_ctrl, &ctrl->start, 1u, "start");

	uint64_t host_ts_begin = 0;
	uint64_t gpu_ts_begin = 0;
	zeDeviceGetGlobalTimestamps(ze_dev, &host_ts_begin, &gpu_ts_begin);

	for (uint32_t r = 0; r < rounds; r++) {
		poll_device_u32(q_ctrl, &ctrl->done_a, r + 1u, "done_a");
	}
	poll_device_u32(q_ctrl, &ctrl->done_b, 1u, "done_b");

	evt_a.wait_and_throw();
	evt_b.wait_and_throw();

	uint64_t host_ts_end = 0;
	uint64_t gpu_ts_end = 0;
	zeDeviceGetGlobalTimestamps(ze_dev, &host_ts_end, &gpu_ts_end);

	uint32_t final_ctr = 0;
	q_b.memcpy(&final_ctr, &dev_b->ctr, sizeof(uint32_t)).wait();
	if (final_ctr != 2u * rounds) {
		sycl::free(dev_a, q_a);
		sycl::free(dev_b, q_b);
		sycl::free(ctrl, q_ctrl);
		throw std::runtime_error("counter check failed in D2D session");
	}

	sycl::free(dev_a, q_a);
	sycl::free(dev_b, q_b);
	sycl::free(ctrl, q_ctrl);

	return double(gpu_ts_end - gpu_ts_begin) * gpu_tick_ns / rounds * 1e-3;
}

int main() {
	sycl::device dev;
	try { dev = sycl::device(sycl::gpu_selector_v); }
	catch (...) { dev = sycl::device(sycl::default_selector_v); }

	sycl::context ctx(dev);
	auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
	const double gpu_tick_ns = get_gpu_tick_ns(ze_dev);

	std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
	std::cout << std::fixed << std::setprecision(1)
			  << "GPU timer: " << gpu_tick_ns << " ns/tick\n";

	std::cout << "Warming up (" << NUM_WARMUP << " rounds)...\n";
	(void)run_session(ctx, dev, NUM_WARMUP, gpu_tick_ns, ze_dev);

	std::cout << "Measuring (" << NUM_ROUNDS << " rounds)...\n";
	const double mean_roundtrip_us =
		run_session(ctx, dev, NUM_ROUNDS, gpu_tick_ns, ze_dev);

	std::cout << std::fixed << std::setprecision(3)
			  << "D2D round-trip mean latency (" << NUM_ROUNDS << " rounds):\n"
			  << "  GPU-timer mean = " << mean_roundtrip_us << " us\n"
			  << "  estimated one-way = " << (mean_roundtrip_us * 0.5) << " us\n";
	return 0;
}
