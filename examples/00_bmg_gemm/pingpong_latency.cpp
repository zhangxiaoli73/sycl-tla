/*
 * PCIe Round-Trip Latency Benchmark
 * ==================================
 * Host (x86-64 inline assembly)  ↔  Intel BMG B60 GPU (inline vISA / DPC++)
 *
 * Set USE_CPP_KERNEL=1 to replace the inline vISA with pure SYCL C++ for
 * protocol validation; set to 0 (default) for the vISA path.
 *
 * Memory layout
 * -------------
 *   device_buf  — zeMemAllocDevice (GPU VRAM, device DRAM).
 *                 The GPU polls its own DRAM (lsc_load.ugm.uc.uc — local, not PCIe).
 *                 The CPU writes via PCIe BAR using MOVNTI+SFENCE (WC write path).
 *                 Two-pointer scheme:
 *                   dev_ptr  — GPU VA (zeMemAllocDevice) passed to the kernel
 *                   cpu_ptr  — host VA (mmap of DMA-BUF export) used by the host
 *                 Both point to the same physical device DRAM pages.
 *                 Investigated approaches that DID NOT work on BMG B60 + xe driver:
 *                   • malloc_device   — GPU-VA-only; no host-page-table entry → segfault
 *                   • malloc_shared + prefetch — shadow copies; concurrent access diverges
 *                   • zeVirtualMemReserve (VMM) — manages GPU VA internally, not host VA
 *                 Solution: zeMemAllocDevice with ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
 *                 export fd via zeMemGetAllocProperties, mmap fd for CPU access.
 *                 memoryAllocationExportTypes = DMA_BUF confirmed on this device.
 *   host_buf    — malloc_host (pinned host DRAM; GPU writes reply here via PCIe)
 *
 * Protocol (round r, 0-based)
 * ---------------------------
 *   1. Host:  RDTSCP → t_start
 *             MOVNTI  device_buf.ctr = 2*r+1   (odd — host writes via PCIe BAR)
 *             SFENCE                            (flush write-combine buffer → device DRAM)
 *   2. GPU:   lsc_load.ugm.uc.uc  poll  device_buf.ctr  until == 2*r+1  (local DRAM poll)
 *             lsc_fence.ugm.invalidate.tile
 *             lsc_store.ugm.uc.uc  host_buf.ctr = 2*r+2  (even — GPU writes via PCIe, posted)
 *             lsc_fence.ugm.evict.tile
 *   3. Host:  PAUSE-loop on host_buf.ctr until == 2*r+2
 *             LFENCE + RDTSCP → t_end
 *             host_sum += (t_end - t_start) * tsc_ns_per_tick
 *   After N rounds: host_mean_us = host_sum / N
 *
 * GPU-timer mean
 * --------------
 * Two zeDeviceGetGlobalTimestamps snapshots bracket the entire N-round session
 * (one after "kernel ready", one after the last reply).  The 27 µs overhead per
 * call is amortised over N rounds (~5 ns/round for N=10,000 — negligible).
 * gpu_mean_us = (gpu_end_tick − gpu_start_tick) × gpu_tick_ns / N / 1000
 *
 * GPU kernel design
 * -----------------
 * A single persistent kernel is launched once for all N rounds.  Re-launching
 * per-round would add tens of µs of kernel-launch overhead, completely swamping
 * the ~1-4 µs PCIe round-trip.
 *
 * The N-round loop is a C++ for-loop.  Inside each iteration, four small asm
 * volatile blocks handle the poll and reply using typed "rw" operand constraints
 * (pattern from oneCCL/src/coll/algorithms/utils/rt64.hpp).  The compiler passes
 * C++ pointer variables directly as vISA register operands — no SLM staging,
 * no .decl boilerplate, and no vISA goto needed.  The C++ while loop handles
 * poll-loop control flow, which IGC compiles correctly.
 *
 * Inline vISA notes (BMG / Xe2)
 * ------------------------------
 * - Typed operand constraints ("rw", "=rw") pass C++ variables as vISA registers
 *   via %0/%1 substitution.  The compiler allocates registers and broadcasts
 *   scalar addresses to SIMD16 automatically.
 * - All UGM (global/PCIe) LSC ops use (M1,16) — SIMD1 ops are silently dropped.
 * - lsc_store.ugm.uc.uc (M1,16) for GPU→host DRAM writes: posted write, GPU does
 *   not wait for a PCIe completion TLP — lower GPU-side reply latency than atomic_store.
 * - %%null is NOT needed for lsc_store (no result operand), unlike lsc_atomic_store.
 * - No sycl::atomic_ref or sycl::atomic_fence anywhere in the kernel — all
 *   system-scope operations go through typed asm UGM instructions directly.
 * - Priming store (lsc_store.ugm.uc.uc to host_buf.ctr=0 before the ready
 *   signal) is required on BMG; without it the first reply is silently dropped.
 *
 * Build
 * -----
 *   icpx -fsycl \
 *        -fsycl-targets=spir64_gen \
 *        -Xsycl-target-backend "-device bmg-g21" \
 *        -O2 -std=c++17 \
 *        -o pingpong_latency pingpong_latency.cpp
 */

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <cassert>
#include <sys/mman.h>
#include <unistd.h>

// ============================================================================
// Configuration
// ============================================================================
static constexpr uint32_t NUM_ROUNDS = 10'000u;
static constexpr uint32_t NUM_WARMUP = 200u;

// Set to 1 to use pure SYCL C++ for the GPU loop (protocol validation),
// set to 0 to use the inline vISA path.
#define USE_CPP_KERNEL 0

// ============================================================================
// Buffer structs — one 64-byte cache line each
// ============================================================================

// Physically resident in GPU VRAM (device DRAM).
// GPU polls via lsc_load.ugm.uc.uc — local DRAM access, not PCIe per iteration.
// CPU writes via PCIe BAR (MOVNTI+SFENCE, WC write path, no snoop needed).
// See DeviceHostMapped for the two-pointer scheme (dev_ptr vs cpu_ptr).
struct alignas(64) DeviceBuf {
    uint32_t ctr;        // host→GPU signal counter
    uint8_t  _pad[60];
};

// Physically resident in pinned host DRAM (malloc_host).
// GPU writes even counter values here via PCIe write; also a "kernel ready" flag.
struct alignas(64) HostBuf {
    uint32_t ctr;        // GPU→host reply counter
    uint32_t ready;      // GPU writes 1 here when kernel has started polling
    uint8_t  _pad[56];
};

static_assert(sizeof(DeviceBuf) == 64, "DeviceBuf must be 64 bytes");
static_assert(sizeof(HostBuf)   == 64, "HostBuf must be 64 bytes");

// ============================================================================
// Host-side timing & I/O helpers — x86-64 inline assembly
// ============================================================================

// RDTSCP: serialises prior instructions, then reads the TSC.
// Use for the start timestamp (placed before the first write).
static inline uint64_t rdtscp_start() {
    uint32_t lo, hi;
    __asm__ volatile (
        "rdtscp"
        : "=a"(lo), "=d"(hi)
        :
        : "rcx", "memory"
    );
    return ((uint64_t)hi << 32) | lo;
}

// LFENCE + RDTSCP: LFENCE ensures all prior loads retire before RDTSCP reads.
// Use for the end timestamp (placed after the poll loop returns).
static inline uint64_t rdtscp_end() {
    uint32_t lo, hi;
    __asm__ volatile (
        "lfence\n\t"
        "rdtscp"
        : "=a"(lo), "=d"(hi)
        :
        : "rcx", "memory"
    );
    return ((uint64_t)hi << 32) | lo;
}

// Store to host DRAM + flush + fence for GPU visibility.
// Used when the target is malloc_host (host-pinned) memory.
// For malloc_host on discrete GPU (Intel BMG):
//   • Regular MOV writes to CPU WB cache.
//   • CLFLUSHOPT flushes the dirty line to DRAM and broadcasts a
//     cache-invalidation snoop over PCIe to the GPU's L3.
//   • SFENCE serialises: no later stores reorder before this point.
// Note: NOT used for dev_buf (which is now malloc_device); kept for
// reference in case host-side memory is ever written this way.
static inline void clflushopt_sfence(volatile uint32_t *dst, uint32_t val) {
    *dst = val;  // cached write → CPU WB cache (dirty line)
    __asm__ volatile (
        "clflushopt (%0)\n\t"
        "sfence"
        :
        : "r"(dst)
        : "memory"
    );
}

// MOVNTI + SFENCE for writes to PCIe BAR-mapped device memory (malloc_device).
// Device VRAM is mapped WC (write-combining) through the PCIe BAR.  On Intel
// Arc with ReBAR, the entire VRAM is host-accessible this way.
//   • MOVNTI writes the 32-bit value via the WC path, bypassing all CPU caches.
//   • SFENCE drains the CPU write-combine buffer and issues the PCIe write TLP.
// No CLFLUSHOPT needed — device memory is never in CPU L1/L2/L3 cache, so
// there is nothing to flush.  The GPU kernel uses lsc_load.ugm.uc.uc which
// bypasses GPU L3 and reads directly from DRAM, so no GPU-side snoop is needed
// either (this was the root cause of the earlier malloc_host MOVNTI problem).
static inline void movnti_sfence(volatile uint32_t *dst, uint32_t val) {
    __asm__ volatile (
        "movnti %1, (%0)\n\t"
        "sfence"
        :
        : "r"(dst), "r"(val)
        : "memory"
    );
}

// Polling spin loop with PAUSE + CLFLUSHOPT.
// CLFLUSHOPT invalidates the CPU cache line before each read so that
// GPU writes to DRAM (via PCIe) are visible even when the PCIe write
// TLP carries "no-snoop", i.e. does not invalidate the CPU L1/L2/L3.
static uint32_t pause_poll(const volatile uint32_t *src, uint32_t expected,
                           const char *tag = "") {
    using clk = std::chrono::steady_clock;
    auto deadline = clk::now() + std::chrono::seconds(2);
    bool warned = false;
    for (;;) {
        // Flush + invalidate the cache line so we see the latest DRAM value.
        __asm__ volatile ("clflushopt (%0)" :: "r"(src) : "memory");
        uint32_t v;
        __asm__ volatile ("movl (%1), %0" : "=r"(v) : "r"(src) : "memory");
        if (v == expected) return v;
        __asm__ volatile ("pause" ::: "memory");
        if (!warned && clk::now() > deadline) {
            fprintf(stderr, "[poll timeout] %s: addr=%p expected=%u got=%u\n",
                    tag, (const void*)src, expected, v);
            fflush(stderr);
            warned = true;
            deadline += std::chrono::seconds(10); // don't spam
        }
    }
}

// ============================================================================
// TSC calibration — uses CLOCK_MONOTONIC to determine ns per TSC tick
// ============================================================================
static double calibrate_tsc_ns() {
    using namespace std::chrono;
    auto c0 = steady_clock::now();
    uint64_t t0 = rdtscp_start();
    std::this_thread::sleep_for(milliseconds(200));
    uint64_t t1 = rdtscp_end();
    auto c1 = steady_clock::now();
    double wall_ns = duration_cast<nanoseconds>(c1 - c0).count();
    return wall_ns / (double)(t1 - t0);
}

// ============================================================================
// DMA-BUF device allocation — device-DRAM-resident, host-writable buffer
// ============================================================================
// Allocate device memory with DMA-BUF export, then mmap it into host address
// space.  This gives two pointers to the same physical device DRAM pages:
//   dev_ptr  — GPU VA (from zeMemAllocDevice); passed to the kernel
//   cpu_ptr  — host VA (from mmap of DMA-BUF fd); used by host MOVNTI writes
//
// Memory access paths:
//   CPU writes:  MOVNTI → WC buffer → SFENCE → PCIe write TLP → device DRAM
//   GPU reads:   lsc_load.ugm.uc.uc → device DRAM (bypasses GPU L3 entirely)
//   No coherency protocol needed: GPU uses uc.uc so it never caches device DRAM;
//   CPU writes go directly to device DRAM via PCIe BAR (WC-mapped by mmap).
struct DeviceHostMapped {
    DeviceBuf           *dev_ptr;    // GPU VA — for kernel lsc_load.ugm.uc.uc
    DeviceBuf           *cpu_ptr;    // host VA (BAR-mapped WC) — for CPU MOVNTI writes
    int                  dma_fd;     // DMA-BUF fd (for cleanup)
    size_t               size;
    ze_context_handle_t  ze_ctx;

    void destroy() {
        if (cpu_ptr) { munmap(cpu_ptr, size); cpu_ptr = nullptr; }
        if (dma_fd >= 0) { close(dma_fd); dma_fd = -1; }
        if (dev_ptr) { zeMemFree(ze_ctx, dev_ptr); dev_ptr = nullptr; }
    }
};

static DeviceHostMapped alloc_device_host_mapped(sycl::queue &kq) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kq.get_context());
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kq.get_device());

    // Allocate device memory with DMA-BUF export capability.
    ze_external_memory_export_desc_t export_desc{
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC, nullptr,
        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF
    };
    ze_device_mem_alloc_desc_t dev_desc{
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &export_desc, 0, 0
    };
    constexpr size_t sz = sizeof(DeviceBuf);
    void *dev_ptr = nullptr;
    ze_result_t rc = zeMemAllocDevice(ze_ctx, &dev_desc, sz, 64, ze_dev, &dev_ptr);
    if (rc != ZE_RESULT_SUCCESS || !dev_ptr)
        throw std::runtime_error("zeMemAllocDevice (DMA-BUF) failed: rc=" + std::to_string(rc));

    // Retrieve the DMA-BUF file descriptor.
    ze_external_memory_export_fd_t export_fd{
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD, nullptr,
        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, -1
    };
    ze_memory_allocation_properties_t mem_props{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
    mem_props.pNext = &export_fd;
    ze_device_handle_t alloc_dev = nullptr;
    rc = zeMemGetAllocProperties(ze_ctx, dev_ptr, &mem_props, &alloc_dev);
    if (rc != ZE_RESULT_SUCCESS || export_fd.fd < 0) {
        zeMemFree(ze_ctx, dev_ptr);
        throw std::runtime_error("zeMemGetAllocProperties (DMA-BUF fd) failed");
    }

    // Map the DMA-BUF into host address space via the PCIe BAR.
    void *cpu_ptr = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, export_fd.fd, 0);
    if (cpu_ptr == MAP_FAILED) {
        close(export_fd.fd);
        zeMemFree(ze_ctx, dev_ptr);
        throw std::runtime_error("mmap of DMA-BUF failed");
    }

    return { static_cast<DeviceBuf*>(dev_ptr),
             static_cast<DeviceBuf*>(cpu_ptr),
             export_fd.fd, sz, ze_ctx };
}
// ============================================================================
// GPU timer calibration — correlates host TSC with GPU global timer
// ============================================================================
// The GPU has a fixed-frequency global timer (timerResolution ns/tick, typically
// 52 ns/tick ≈ 19 MHz on BMG B60).  zeDeviceGetGlobalTimestamps reads both the
// host clock and GPU clock simultaneously (or as close as the hardware allows).
// We call it N times and average to reduce noise.
//
// Returns a GpuTimerCal that lets you convert a GPU tick to a host TSC value:
//   host_tsc_equiv = gpu_tick * cal.gpu_ticks_to_host_tsc + cal.gpu_host_tsc_offset
//
// This is the foundation for GPU-GPU latency measurement: calibrate each GPU
// against the host, then align both GPU clocks to the same host timeline.
struct GpuTimerCal {
    double   gpu_tick_ns;            // GPU timer resolution in ns (from timerResolution)
    double   lz_call_overhead_ns;    // measured overhead of zeDeviceGetGlobalTimestamps call
    double   epoch_diff_ns;          // gpu_tick * gpu_tick_ns + epoch_diff_ns == host LZ ns
    // Note on one-way latency:
    //   Computing per-round GPU→host one-way latency requires the GPU timer value AT
    //   THE MOMENT the GPU sends each reply (inside the persistent kernel).  No public
    //   API or known vISA instruction currently provides this from within a regular SYCL
    //   kernel on Xe2/BMG.  The fields above enable future GPU-GPU work where each GPU
    //   would write its own in-kernel timer to a shared buffer, and epoch_diff_ns aligns
    //   the two GPU clocks to a common host timeline.
    //
    // Unused fields kept for API consistency:
    double   gpu_ticks_to_host_tsc;  // (not meaningful — h_lz is in ns, not TSC)
    int64_t  gpu_host_tsc_offset;    // (not meaningful — kept for future TSC-based variant)
};

static GpuTimerCal calibrate_gpu_host(sycl::queue &kq, double tsc_ns) {
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                      kq.get_device());

    // Query GPU timer resolution (ns per tick).
    ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    zeDeviceGetProperties(ze_dev, &props);
    double gpu_tick_ns = static_cast<double>(props.timerResolution);

    // Measure the overhead of a zeDeviceGetGlobalTimestamps call via TSC bracket.
    // Also collect (host_ns_from_lz, gpu_tick) pairs to establish the epoch offset.
    static constexpr int N = 20;
    double sum_lz_overhead_ns = 0.0;
    double sum_epoch_diff_ns  = 0.0;  // h_lz_ns - g_lz * gpu_tick_ns
    for (int i = 0; i < N; i++) {
        uint64_t t0 = rdtscp_start();
        uint64_t h_lz = 0, g_lz = 0;
        zeDeviceGetGlobalTimestamps(ze_dev, &h_lz, &g_lz);
        uint64_t t1 = rdtscp_end();
        // h_lz is in nanoseconds (Level Zero host timestamp, LZ epoch).
        // Estimate call overhead in ns using TSC bracket.
        double call_overhead_ns = (double)(t1 - t0) * tsc_ns;
        sum_lz_overhead_ns += call_overhead_ns;
        // Epoch difference: offset between LZ host ns epoch and GPU timer ns epoch.
        sum_epoch_diff_ns  += (double)h_lz - (double)g_lz * gpu_tick_ns;
    }

    GpuTimerCal cal;
    cal.gpu_tick_ns          = gpu_tick_ns;
    cal.gpu_ticks_to_host_tsc = 0.0;  // not used — see note below
    cal.gpu_host_tsc_offset  = 0;

    // Store derived quantities for reporting.
    // lz_call_overhead_ns: time between rdtscp and zeDeviceGetGlobalTimestamps return
    cal.lz_call_overhead_ns = sum_lz_overhead_ns / N;
    // epoch_diff_ns: constant offset such that gpu_tick * gpu_tick_ns + epoch_diff_ns
    //               gives the host LZ ns equivalent of a GPU timer tick.
    cal.epoch_diff_ns = sum_epoch_diff_ns / N;
    return cal;
}

// ============================================================================
static void submit_pingpong_kernel(sycl::queue &kq,
                                   DeviceBuf   *dev_ptr,
                                   HostBuf     *host_ptr,
                                   uint32_t     num_rounds)
{
    kq.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(16)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]]
            {
                if (item.get_local_id(0) != 0) return;

                uint32_t *dp = &dev_ptr->ctr;
                uint32_t *hp = &host_ptr->ctr;
                uint32_t *rp = &host_ptr->ready;

                // PCIe-path priming: write 0 to host_ptr->ctr via the same
                // lsc_store.ugm.uc.uc + evict-fence used by the main loop.
                // This establishes the PCIe write path so subsequent asm UGM
                // stores in the main loop are not silently dropped on BMG.
                {
                    uint32_t zero = 0u;
#ifdef __SYCL_DEVICE_ONLY__
                    __asm__ volatile (
                        "lsc_store.ugm.uc.uc (M1, 16)"
                        "  flat[%0]:a64 %1:d32\n"
                        : : "rw"(hp), "rw"(zero)
                    );
                    __asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
                    *hp = zero;
#endif
                }

                // Signal host: kernel is running and ready for round 0.
                {
                    uint32_t one = 1u;
#ifdef __SYCL_DEVICE_ONLY__
                    __asm__ volatile (
                        "lsc_store.ugm.uc.uc (M1, 16)"
                        "  flat[%0]:a64 %1:d32\n"
                        : : "rw"(rp), "rw"(one)
                    );
                    __asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
                    *rp = one;
#endif
                }

#if USE_CPP_KERNEL
                // ── Pure SYCL C++ path (protocol validation) ──────────────────
                for (uint32_t r = 0; r < num_rounds; r++) {
                    const uint32_t exp  = 2u * r + 1u;
                    const uint32_t rply = exp + 1u;
                    volatile uint32_t *vdp = dp;
                    volatile uint32_t *vhp = hp;
                    while (*vdp != exp) {}
                    *vhp = rply;
                }
#else
                // ── Inline vISA path ───────────────────────────────────────────
                //
                // Typed "rw" constraints pass C++ pointer variables directly as
                // vISA register operands (%0, %1 substitution).  The compiler
                // handles register allocation and SIMD16 address broadcast.
                //
                // Per-round structure:
                //   1. C++ while loop: asm load polls dev_buf.ctr (device DRAM, local) until == exp
                //   2. asm fence (invalidate) — ensure load ordering
                //   3. asm store (posted)     — write reply to host_buf.ctr via PCIe
                //   4. asm fence (evict)      — push write out across PCIe to host DRAM
                //
                for (uint32_t r = 0; r < num_rounds; r++) {
                    const uint32_t exp = 2u * r + 1u;
                    const uint32_t rep = 2u * r + 2u;

                    uint32_t val;
                    do {
#ifdef __SYCL_DEVICE_ONLY__
                        __asm__ volatile (
                            "lsc_load.ugm.uc.uc (M1, 16) %0:d32 flat[%1]:a64\n"
                            : "=rw"(val) : "rw"(dp)
                        );
#else
                        val = *dp;
#endif
                    } while (val != exp);

#ifdef __SYCL_DEVICE_ONLY__
                    __asm__ volatile ("lsc_fence.ugm.invalidate.tile\n" : : :);
                    __asm__ volatile (
                        "lsc_store.ugm.uc.uc (M1, 16)"
                        "  flat[%0]:a64 %1:d32\n"
                        : : "rw"(hp), "rw"(rep)
                    );
                    __asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
                    *hp = rep;
#endif
                }
#endif // USE_CPP_KERNEL

                // Final fence: ensure all writes are visible before kernel exits.
#ifdef __SYCL_DEVICE_ONLY__
                __asm__ volatile ("lsc_fence.ugm.evict.gpu\n" : : :);
#endif
            }
        );
    });
}
// ============================================================================
// Statistics helpers
// ============================================================================
// Session result: mean RTT from both the host TSC and the GPU global timer.
// Two measurements of the same wall-clock interval using independent clocks;
// comparing them cross-validates clock accuracy and epoch alignment.
struct SessionResult {
    double host_mean_us;   // mean RTT measured by host TSC
    double gpu_mean_us;    // mean RTT measured by GPU global timer (via zeDeviceGetGlobalTimestamps)
};

// ============================================================================
// One measurement session
// ============================================================================
// GPU-side mean RTT:
//   Two zeDeviceGetGlobalTimestamps snapshots bracket the entire N-round session.
//   The 27 µs call overhead per snapshot is amortised over N rounds, adding only
//   ~5 ns/round to the uncertainty for N=10,000 — completely negligible.
//   The GPU timer (fixed-frequency, 52 ns/tick on BMG B60) is independent of the
//   host TSC; agreement between the two means cross-validates both clocks.
static SessionResult run_session(sycl::queue       &kq,
                                 DeviceBuf         *dev_ptr,   // GPU VA — for kernel
                                 DeviceBuf         *cpu_ptr,   // host VA — for CPU MOVNTI writes
                                 HostBuf           *host_buf,
                                 uint32_t           rounds,
                                 double             tsc_ns,
                                 const GpuTimerCal &gpu_cal,
                                 ze_device_handle_t ze_dev)
{
    // Zero both buffers before each session.
    // dev_ptr is a raw LZ allocation (not SYCL-tracked); use MOVNTI via cpu_ptr.
    movnti_sfence(&cpu_ptr->ctr, 0u);
    std::memset(host_buf, 0, sizeof(HostBuf));

    // Submit the persistent kernel — it will spin-wait for the first signal.
    submit_pingpong_kernel(kq, dev_ptr, host_buf, rounds);

    // Wait until the kernel has started and written ready = 1 to host_buf.
    pause_poll(&host_buf->ready, 1u, "kernel-ready");

    // Host→device PCIe path priming: one MOVNTI write before the main loop.
    // Analogous to the GPU→host priming done inside the kernel.
    // Primes the PCIe write path so the first round's signal is not dropped.
    movnti_sfence(&cpu_ptr->ctr, 0u);

    // GPU timer snapshot A: kernel is ready, round 0 is about to begin.
    uint64_t h_lz_start = 0, gpu_start_tick = 0;
    zeDeviceGetGlobalTimestamps(ze_dev, &h_lz_start, &gpu_start_tick);

    double host_sum_us = 0.0;

    for (uint32_t r = 0; r < rounds; r++) {
        const uint32_t signal_val = 2u * r + 1u;   // odd  — host fires
        const uint32_t reply_val  = 2u * r + 2u;   // even — GPU replies

        uint64_t t_start = rdtscp_start();

        // MOVNTI + SFENCE: WC BAR write → PCIe write TLP → device DRAM.
        movnti_sfence(&cpu_ptr->ctr, signal_val);

        // Spin on host_buf until GPU writes the reply.
        pause_poll(&host_buf->ctr, reply_val, "gpu-reply");

        // End timer (LFENCE ensures the poll load has retired).
        uint64_t t_end = rdtscp_end();

        host_sum_us += double(t_end - t_start) * tsc_ns * 1e-3;
    }

    // GPU timer snapshot B: last reply has been detected, session is over.
    uint64_t h_lz_end = 0, gpu_end_tick = 0;
    zeDeviceGetGlobalTimestamps(ze_dev, &h_lz_end, &gpu_end_tick);

    // Wait for the kernel to exit cleanly.
    kq.wait_and_throw();

    SessionResult res;
    res.host_mean_us = host_sum_us / rounds;
    // GPU mean: total GPU-timer ticks × ns/tick ÷ rounds ÷ 1000 (ns→µs).
    res.gpu_mean_us  = double(gpu_end_tick - gpu_start_tick)
                       * gpu_cal.gpu_tick_ns / rounds * 1e-3;
    return res;
}

// ============================================================================
// main
// ============================================================================
int main()
{
    // ── Device & queue ────────────────────────────────────────────────────────
    sycl::device dev;
    try { dev = sycl::device(sycl::gpu_selector_v); }
    catch (...) { dev = sycl::device(sycl::default_selector_v); }
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";

    sycl::context ctx(dev);
    sycl::queue kq(ctx, dev, sycl::property::queue::in_order{});

    // Get Level Zero device handle for GPU timer calls.
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);

    // ── USM allocations ───────────────────────────────────────────────────────
    // dev_buf: zeMemAllocDevice + DMA-BUF mmap — device DRAM, two-pointer scheme.
    //   dev_mapped.dev_ptr — GPU VA for the kernel
    //   dev_mapped.cpu_ptr — host VA (PCIe BAR, WC-mapped) for MOVNTI writes
    DeviceHostMapped dev_mapped = alloc_device_host_mapped(kq);
    auto *host_buf = sycl::malloc_host<HostBuf>(1, kq);
    if (!host_buf) {
        std::cerr << "USM allocation failed\n";
        dev_mapped.destroy();
        return 1;
    }
    std::cout << "dev_buf: zeMemAllocDevice + DMA-BUF mmap (device DRAM)\n"
              << "  GPU VA (kernel ptr): " << dev_mapped.dev_ptr << "\n"
              << "  CPU VA (host ptr):   " << dev_mapped.cpu_ptr << "\n";

    // ── Calibrate TSC ─────────────────────────────────────────────────────────
    std::cout << "Calibrating TSC...\n";
    const double tsc_ns = calibrate_tsc_ns();
    std::cout << "TSC: " << std::fixed << std::setprecision(3)
              << (1000.0 / tsc_ns) << " MHz  (" << tsc_ns << " ns/tick)\n";

    // ── Calibrate GPU timer ───────────────────────────────────────────────────
    std::cout << "Calibrating GPU timer...\n";
    const GpuTimerCal gpu_cal = calibrate_gpu_host(kq, tsc_ns);
    std::cout << "GPU timer: " << std::fixed << std::setprecision(1)
              << gpu_cal.gpu_tick_ns << " ns/tick  ("
              << std::setprecision(0) << (1e3 / gpu_cal.gpu_tick_ns) << " MHz)"
              << "  zeDeviceGetGlobalTimestamps overhead: "
              << std::setprecision(2) << gpu_cal.lz_call_overhead_ns * 1e-3 << " µs\n";
    std::cout << "  GPU-host epoch aligned (offset="
              << std::setprecision(3) << gpu_cal.epoch_diff_ns * 1e-9 << " s)\n"
              << "  Note: per-round one-way latency requires in-kernel GPU timer access\n"
              << "        (no public vISA instruction found for Xe2/BMG in SYCL kernels)\n"
              << "        Future GPU-GPU work: each GPU writes in-kernel timer to shared\n"
              << "        buffer; use epoch_diff to align both GPU clocks to host.\n";

    // ── Warmup ────────────────────────────────────────────────────────────────
    std::cout << "Warming up (" << NUM_WARMUP << " rounds)...\n";
    run_session(kq, dev_mapped.dev_ptr, dev_mapped.cpu_ptr, host_buf, NUM_WARMUP, tsc_ns, gpu_cal, ze_dev);

    // ── Measured run ──────────────────────────────────────────────────────────
    std::cout << "Measuring (" << NUM_ROUNDS << " rounds)...\n";
    auto result = run_session(kq, dev_mapped.dev_ptr, dev_mapped.cpu_ptr, host_buf, NUM_ROUNDS, tsc_ns, gpu_cal, ze_dev);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "PCIe round-trip mean latency (" << NUM_ROUNDS << " rounds):\n"
              << "  host-TSC mean = " << result.host_mean_us << " µs\n"
              << "  GPU-timer mean = " << result.gpu_mean_us  << " µs"
              << "  (GPU timer: " << std::setprecision(1) << gpu_cal.gpu_tick_ns << " ns/tick"
              << ", snapshot overhead amortised: "
              << std::setprecision(3) << gpu_cal.lz_call_overhead_ns * 2e-3 / NUM_ROUNDS
              << " µs/round)\n";

    // ── Verify final counter ──────────────────────────────────────────────────
    const uint32_t expected_final = 2u * NUM_ROUNDS;
    if (host_buf->ctr != expected_final) {
        std::cerr << "WARN: host_buf->ctr = " << host_buf->ctr
                  << "  expected " << expected_final
                  << "  (possible synchronisation issue)\n";
    } else {
        std::cout << "Counter check OK  (host_buf->ctr = " << host_buf->ctr << ")\n";
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    dev_mapped.destroy();     // munmap + close(dma_fd) + zeMemFree
    sycl::free(host_buf, kq);
    return 0;
}
