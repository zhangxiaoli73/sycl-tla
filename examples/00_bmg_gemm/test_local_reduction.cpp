/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*
 * Unit test for Phase 3 local reduction kernel (sycl::vec<bfloat16> vectorized sum of 4 buffers).
 *
 * Build:
 *   ninja test_local_reduction
 *
 * Run:
 *   ./test_local_reduction
 */

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using BF16 = sycl::ext::oneapi::bfloat16;

constexpr int64_t ROWS = 2048;
constexpr int64_t COLS = 4096;
constexpr int64_t N_ELEMS = ROWS * COLS;   // 8M elements
constexpr int NUM_BUFS = 4;
constexpr int WARMUP_ITERS = 10;
constexpr int BENCH_ITERS  = 50;

// ── Templatized kernel (NUM_PER_TH as template param) ───────────────────────
template <int NUM_PER_TH>
struct ReductionKernel;   // kernel name tag

// NUM_PER_TH = number of int64_t loads per work-item
// Each int64_t = 4 bf16, so each work-item processes NUM_PER_TH * 4 bf16 elements
template <int NUM_PER_TH>
sycl::event launch_local_reduction(
		sycl::queue& q,
		const BF16* buf0_ptr,
		const BF16* buf1_ptr,
		const BF16* buf2_ptr,
		const BF16* buf3_ptr,
		BF16* out_ptr,
		int64_t n_elems) {

	constexpr int BF16_PER_I64 = 4;  // 64bit / 16bit
	const int WG_SIZE = static_cast<int>(q.get_device().get_info<sycl::info::device::max_work_group_size>())/NUM_PER_TH;

	const int64_t i64_elems = n_elems / BF16_PER_I64;
	const int64_t TILE_SIZE = static_cast<int64_t>(WG_SIZE) * NUM_PER_TH;
	const int64_t n_groups = (i64_elems + TILE_SIZE - 1) / TILE_SIZE;

	auto out_i64 = reinterpret_cast<int64_t*>(out_ptr);
	auto b0 = reinterpret_cast<const int64_t*>(buf0_ptr);
	auto b1 = reinterpret_cast<const int64_t*>(buf1_ptr);
	auto b2 = reinterpret_cast<const int64_t*>(buf2_ptr);
	auto b3 = reinterpret_cast<const int64_t*>(buf3_ptr);

	return q.submit([&](sycl::handler& h) {
		h.parallel_for<ReductionKernel<NUM_PER_TH>>(
			sycl::nd_range<1>(
				sycl::range<1>(n_groups * WG_SIZE),
				sycl::range<1>(WG_SIZE)),
			[=](sycl::nd_item<1> item)
			[[sycl::reqd_sub_group_size(16)]] {
				const int64_t wi_base = static_cast<int64_t>(item.get_global_linear_id()) * NUM_PER_TH;

				for (int v = 0; v < NUM_PER_TH; ++v) {
					const int64_t idx = wi_base + v;
					if (idx >= i64_elems) break;

					// 64-bit load from each buffer
					int64_t raw0 = b0[idx];
					int64_t raw1 = b1[idx];
					int64_t raw2 = b2[idx];
					int64_t raw3 = b3[idx];

					// Unpack 4 bf16, sum in float, repack
					int64_t result = 0;
					for (int k = 0; k < BF16_PER_I64; ++k) {
						int shift = k * 16;
						uint16_t h0 = static_cast<uint16_t>((raw0 >> shift) & 0xFFFF);
						uint16_t h1 = static_cast<uint16_t>((raw1 >> shift) & 0xFFFF);
						uint16_t h2 = static_cast<uint16_t>((raw2 >> shift) & 0xFFFF);
						uint16_t h3 = static_cast<uint16_t>((raw3 >> shift) & 0xFFFF);
						BF16 f0 = sycl::bit_cast<BF16>(h0);
						BF16 f1 = sycl::bit_cast<BF16>(h1);
						BF16 f2 = sycl::bit_cast<BF16>(h2);
						BF16 f3 = sycl::bit_cast<BF16>(h3);
						BF16 sum = static_cast<BF16>(
							float(f0) + float(f1) + float(f2) + float(f3));
						uint16_t packed = sycl::bit_cast<uint16_t>(sum);
						result |= (static_cast<int64_t>(packed) << shift);
					}
					out_i64[idx] = result;
				}
			});
	});
}

// ── Correctness test ────────────────────────────────────────────────────────
template <int NUM_PER_TH>
bool run_correctness(sycl::queue& q, const BF16* d_bufs[NUM_BUFS], BF16* d_out,
		const std::vector<BF16> h_bf16_bufs[NUM_BUFS]) {

	q.memset(d_out, 0, N_ELEMS * sizeof(BF16)).wait();
	launch_local_reduction<NUM_PER_TH>(q, d_bufs[0], d_bufs[1], d_bufs[2], d_bufs[3], d_out, N_ELEMS);
	q.wait();

	std::vector<BF16> h_result(N_ELEMS);
	q.memcpy(h_result.data(), d_out, N_ELEMS * sizeof(BF16)).wait();

	double max_abs_diff = 0.0;
	size_t mismatch = 0;
	for (int64_t i = 0; i < N_ELEMS; ++i) {
		float ref = 0.0f;
		for (int b = 0; b < NUM_BUFS; ++b) {
			ref += static_cast<float>(h_bf16_bufs[b][i]);
		}
		float val = static_cast<float>(h_result[i]);
		double diff = std::abs(static_cast<double>(ref) - static_cast<double>(val));
		max_abs_diff = std::max(max_abs_diff, diff);
		double rel = (std::abs(ref) > 1e-6f) ? diff / std::abs(ref) : diff;
		if (rel > 5e-2) mismatch++;
	}

	bool passed = (mismatch == 0);
	printf("  [correctness NUM_PER_TH=%d] max_abs=%.6e  mismatches=%zu/%ld  %s\n",
		NUM_PER_TH, max_abs_diff, mismatch, static_cast<long>(N_ELEMS),
		passed ? "PASSED" : "FAILED");
	return passed;
}

// ── Benchmark ───────────────────────────────────────────────────────────────
template <int NUM_PER_TH>
double run_bench(sycl::queue& q, const BF16* d_bufs[NUM_BUFS], BF16* d_out) {
	// warmup
	for (int i = 0; i < WARMUP_ITERS; ++i) {
		launch_local_reduction<NUM_PER_TH>(q, d_bufs[0], d_bufs[1], d_bufs[2], d_bufs[3], d_out, N_ELEMS);
	}
	q.wait();

	// benchmark with event profiling
	std::vector<sycl::event> events(BENCH_ITERS);
	for (int i = 0; i < BENCH_ITERS; ++i) {
		events[i] = launch_local_reduction<NUM_PER_TH>(q, d_bufs[0], d_bufs[1], d_bufs[2], d_bufs[3], d_out, N_ELEMS);
	}
	q.wait();

	double total_ns = 0.0;
	for (int i = 0; i < BENCH_ITERS; ++i) {
		auto start = events[i].get_profiling_info<sycl::info::event_profiling::command_start>();
		auto end   = events[i].get_profiling_info<sycl::info::event_profiling::command_end>();
		total_ns += static_cast<double>(end - start);
	}
	double avg_us = total_ns / BENCH_ITERS / 1000.0;

	// bandwidth: 4 reads + 1 write, each N_ELEMS * 2 bytes (bf16)
	double bytes = static_cast<double>(N_ELEMS) * sizeof(BF16) * (NUM_BUFS + 1);
	double bw_gb_s = bytes / (avg_us * 1e-6) / 1e9;

	printf("  [bench NUM_PER_TH=%2d] avg=%.2f us  BW=%.2f GB/s\n",
		NUM_PER_TH, avg_us, bw_gb_s);
	return avg_us;
}

// ── Run one NUM_PER_TH variant (correctness + bench) ────────────────────────
template <int NUM_PER_TH>
bool run_variant(sycl::queue& q, const BF16* d_bufs[NUM_BUFS], BF16* d_out,
		const std::vector<BF16> h_bf16_bufs[NUM_BUFS]) {
	bool ok = run_correctness<NUM_PER_TH>(q, d_bufs, d_out, h_bf16_bufs);
	run_bench<NUM_PER_TH>(q, d_bufs, d_out);
	return ok;
}

int main() {
	sycl::queue q{sycl::gpu_selector_v,
		sycl::property_list{sycl::property::queue::in_order{},
		                    sycl::property::queue::enable_profiling{}}};

	auto dev = q.get_device();
	printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
	printf("Max work-group size: %zu\n",
		dev.get_info<sycl::info::device::max_work_group_size>());
	printf("Buffer shape: %ldx%ld = %ld elements (%.2f MiB bf16)\n",
		static_cast<long>(ROWS), static_cast<long>(COLS),
		static_cast<long>(N_ELEMS),
		static_cast<double>(N_ELEMS * sizeof(BF16)) / (1024.0 * 1024.0));

	// Prepare host data
	std::vector<BF16> h_bf16_bufs[NUM_BUFS];
	for (int b = 0; b < NUM_BUFS; ++b) {
		h_bf16_bufs[b].resize(N_ELEMS);
		for (int64_t i = 0; i < N_ELEMS; ++i) {
			h_bf16_bufs[b][i] = static_cast<BF16>((b + 1) * 0.25f + (i % 1024) * 0.001f);
		}
	}

	// Device allocations
	const BF16* d_bufs[NUM_BUFS];
	for (int b = 0; b < NUM_BUFS; ++b) {
		BF16* p = sycl::malloc_device<BF16>(N_ELEMS, q);
		q.memcpy(p, h_bf16_bufs[b].data(), N_ELEMS * sizeof(BF16));
		d_bufs[b] = p;
	}
	BF16* d_out = sycl::malloc_device<BF16>(N_ELEMS, q);
	q.wait();

	printf("\n── Correctness + Benchmark (sweep NUM_PER_TH) ──\n");
	bool all_passed = true;
	all_passed &= run_variant<1>(q, d_bufs, d_out, h_bf16_bufs);
	all_passed &= run_variant<2>(q, d_bufs, d_out, h_bf16_bufs);
	all_passed &= run_variant<4>(q, d_bufs, d_out, h_bf16_bufs);
	all_passed &= run_variant<8>(q, d_bufs, d_out, h_bf16_bufs);
	all_passed &= run_variant<16>(q, d_bufs, d_out, h_bf16_bufs);

	// Cleanup
	for (int b = 0; b < NUM_BUFS; ++b)
		sycl::free(const_cast<BF16*>(d_bufs[b]), q);
	sycl::free(d_out, q);

	printf("\n=== Overall: %s ===\n", all_passed ? "ALL PASSED" : "SOME FAILED");
	return all_passed ? 0 : 1;
}
