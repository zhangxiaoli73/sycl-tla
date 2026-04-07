#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "symm.hpp"

// ---------------------------------------------------------------------------
// Test 1: SymmMemory construction & basic accessors
// ---------------------------------------------------------------------------
static bool test_construction(sycl::queue& q, int rank, int world_size) {
    std::cout << "[rank " << rank << "] test_construction ..." << std::endl;

    int m = 64, n = 32, k = 16;
    int num_channels = 8;

    SymmMemory symm(m, n, k, rank, world_size, q, num_channels);

    if (symm.get_m() != m || symm.get_n() != n || symm.get_k() != k) {
        std::cerr << "[rank " << rank << "] FAIL: accessor mismatch" << std::endl;
        return false;
    }
    if (symm.get_rank() != rank || symm.get_world_size() != world_size) {
        std::cerr << "[rank " << rank << "] FAIL: rank/world_size mismatch" << std::endl;
        return false;
    }
    if (symm.local_data_ptr_ == nullptr) {
        std::cerr << "[rank " << rank << "] FAIL: local_data_ptr_ is null" << std::endl;
        return false;
    }
    if (symm.local_signal_ptr_ == nullptr) {
        std::cerr << "[rank " << rank << "] FAIL: local_signal_ptr_ is null" << std::endl;
        return false;
    }

    std::cout << "[rank " << rank << "] test_construction PASSED" << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
// Test 2: get_data_buffer / get_flag_buffer return non-null for all ranks
// ---------------------------------------------------------------------------
static bool test_ipc_buffers(sycl::queue& q, int rank, int world_size) {
    std::cout << "[rank " << rank << "] test_ipc_buffers ..." << std::endl;

    int m = 64, n = 32, k = 16;
    SymmMemory symm(m, n, k, rank, world_size, q, 8);

    for (int r = 0; r < world_size; ++r) {
        void* data = symm.get_data_buffer(r);
        void* flag = symm.get_flag_buffer(r);
        if (data == nullptr) {
            std::cerr << "[rank " << rank << "] FAIL: get_data_buffer(" << r << ") is null" << std::endl;
            return false;
        }
        if (flag == nullptr) {
            std::cerr << "[rank " << rank << "] FAIL: get_flag_buffer(" << r << ") is null" << std::endl;
            return false;
        }
    }

    std::cout << "[rank " << rank << "] test_ipc_buffers PASSED" << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
// Test 3: barrier synchronization
//   All ranks write their rank value into their local data buffer, then call
//   barrier, then each rank reads every peer's data buffer to verify the
//   write is visible.
// ---------------------------------------------------------------------------
static bool test_barrier(sycl::queue& q, int rank, int world_size) {
    std::cout << "[rank " << rank << "] test_barrier ..." << std::endl;

    int m = 64, n = 32, k = 16;
    SymmMemory symm(m, n, k, rank, world_size, q, 8);

    // Each rank writes a known pattern into its own slice of the data buffer.
    // Data layout: [world_size][m][k] of 16-bit elements.
    size_t shard_elems = static_cast<size_t>(m) * k;
    size_t shard_bytes = shard_elems * 2; // 16-bit

    // Fill local shard (at offset rank * shard_bytes) with rank value
    uint16_t fill_val = static_cast<uint16_t>(rank + 1);
    void* local_data = symm.local_data_ptr_;
    char* local_shard = reinterpret_cast<char*>(local_data) + static_cast<size_t>(rank) * shard_bytes;

    // Use memset-like fill: set all bytes to low byte of fill_val
    // For simplicity, just memset with (rank+1) — gives a recognizable pattern
    q.memset(local_shard, static_cast<int>(fill_val & 0xFF), shard_bytes).wait();

    // Barrier — ensures all ranks have finished writing
    symm.barrier(0, q);
    MPI_Barrier(MPI_COMM_WORLD);

    // Now verify: read remote data buffers to check that peer writes are visible
    for (int peer = 0; peer < world_size; ++peer) {
        void* peer_data = symm.get_data_buffer(peer);
        char* peer_shard = reinterpret_cast<char*>(peer_data) + static_cast<size_t>(peer) * shard_bytes;

        // Copy a few bytes back to host to verify
        constexpr int check_bytes = 16;
        std::vector<uint8_t> host_buf(check_bytes, 0);
        q.memcpy(host_buf.data(), peer_shard, check_bytes).wait();

        uint8_t expected = static_cast<uint8_t>((peer + 1) & 0xFF);
        for (int i = 0; i < check_bytes; ++i) {
            if (host_buf[i] != expected) {
                std::cerr << "[rank " << rank << "] FAIL: peer " << peer
                          << " data mismatch at byte " << i
                          << " expected=" << static_cast<int>(expected)
                          << " got=" << static_cast<int>(host_buf[i]) << std::endl;
                return false;
            }
        }
    }

    std::cout << "[rank " << rank << "] test_barrier PASSED" << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
// Test 4: multiple barriers in sequence
// ---------------------------------------------------------------------------
static bool test_multiple_barriers(sycl::queue& q, int rank, int world_size) {
    std::cout << "[rank " << rank << "] test_multiple_barriers ..." << std::endl;

    int m = 64, n = 32, k = 16;
    SymmMemory symm(m, n, k, rank, world_size, q, 8);

    for (int iter = 0; iter < 5; ++iter) {
        symm.barrier(0, q);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::cout << "[rank " << rank << "] test_multiple_barriers PASSED" << std::endl;
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        if (rank == 0) std::cerr << "No GPU devices found" << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (static_cast<size_t>(rank) >= devices.size()) {
        std::cerr << "Rank " << rank << " has no matching GPU device" << std::endl;
        MPI_Finalize();
        return 1;
    }

    auto dev = devices[rank];
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev,
                  sycl::property_list{sycl::property::queue::in_order{},
                                      sycl::property::queue::enable_profiling{}});

    if (rank == 0) {
        std::cout << "=== SymmMemory tests, world_size=" << world_size << " ===" << std::endl;
    }

    int failures = 0;

    auto run_test = [&](const char* name, bool (*fn)(sycl::queue&, int, int)) {
        MPI_Barrier(MPI_COMM_WORLD);
        bool ok = false;
        try {
            ok = fn(q, rank, world_size);
        } catch (std::exception const& e) {
            std::cerr << "[rank " << rank << "] " << name << " EXCEPTION: " << e.what() << std::endl;
        }
        if (!ok) ++failures;
        MPI_Barrier(MPI_COMM_WORLD);
    };

    run_test("test_construction", test_construction);
    run_test("test_ipc_buffers", test_ipc_buffers);
    run_test("test_barrier", test_barrier);
    run_test("test_multiple_barriers", test_multiple_barriers);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        if (failures == 0) {
            std::cout << "=== ALL TESTS PASSED ===" << std::endl;
        } else {
            std::cout << "=== " << failures << " TEST(S) FAILED ===" << std::endl;
        }
    }

    MPI_Finalize();
    return failures > 0 ? 1 : 0;
}
