#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include <string>
#include <cstdint>
#include <ctime>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#if defined(__linux__)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/initialize_block.hpp"

#include "common/sycl_cute_common.hpp"

using namespace cute;

inline void rs_debug_log(int rank, const std::string& message) {
    std::cerr << "[rank " << rank << "] [debug] " << message << std::endl;
}

#if defined(__linux__)
struct rs_ipc_exchange_fd {
    char storage[CMSG_LEN(sizeof(int)) - sizeof(int)];
    int fd;

    rs_ipc_exchange_fd(int cmsg_level, int cmsg_type, int descriptor) : fd(descriptor) {
        auto* cmsg = reinterpret_cast<cmsghdr*>(storage);
        cmsg->cmsg_len = sizeof(rs_ipc_exchange_fd);
        cmsg->cmsg_level = cmsg_level;
        cmsg->cmsg_type = cmsg_type;
    }

    rs_ipc_exchange_fd() : fd(-1) {
        std::memset(storage, 0, sizeof(storage));
    }
};
#endif

struct rs_ipc_payload {
    int rank;
    uint64_t offset;
};

struct rs_ipc_peer_info {
    int fd = -1;
    size_t offset = 0;
};

inline int rs_next_exchange_id() {
    static int exchange_id = 0;
    return exchange_id++;
}

// MPI datatype helper
template <typename T> MPI_Datatype mpi_type();
template <> MPI_Datatype mpi_type<float>()  { return MPI_FLOAT; }
template <> MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }

// ---------------- Level Zero IPC Helpers ----------------

inline void ze_check_result(ze_result_t res, const char* op, int rank) {
    if (res == ZE_RESULT_SUCCESS) {
        return;
    }
    throw std::runtime_error(
        std::string("[rank ") + std::to_string(rank) + "] " + op +
        " failed, ze_result_t=" + std::to_string(static_cast<int>(res)));
}

inline ze_ipc_mem_handle_t ze_get_ipc_handle(sycl::context const& ctx, void* ptr, int rank) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    ze_ipc_mem_handle_t handle = {};
    auto res = zeMemGetIpcHandle(ze_ctx, ptr, &handle);
    ze_check_result(res, "zeMemGetIpcHandle", rank);
    return handle;
}

inline std::pair<void*, size_t> ze_get_ipc_base_and_offset(
    sycl::context const& ctx,
    void* ptr,
    int rank) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    void* base_addr = nullptr;
    size_t base_size = 0;
    auto res = zeMemGetAddressRange(ze_ctx, ptr, &base_addr, &base_size);
    ze_check_result(res, "zeMemGetAddressRange", rank);
    if (base_addr == nullptr || base_size == 0) {
        throw std::runtime_error(
            std::string("[rank ") + std::to_string(rank) + "] zeMemGetAddressRange returned invalid range");
    }
    auto offset = static_cast<size_t>(reinterpret_cast<char*>(ptr) - reinterpret_cast<char*>(base_addr));
    return {base_addr, offset};
}

inline void* ze_open_ipc_handle(sycl::context const& ctx, sycl::device const& dev,
                                ze_ipc_mem_handle_t handle, int rank, int peer_rank) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    void* ptr = nullptr;
    auto res = zeMemOpenIpcHandle(ze_ctx, ze_dev, handle, 0, &ptr);
    ze_check_result(res, "zeMemOpenIpcHandle", rank);
    if (ptr == nullptr) {
        throw std::runtime_error(
            std::string("[rank ") + std::to_string(rank) + "] zeMemOpenIpcHandle returned nullptr for peer rank " +
            std::to_string(peer_rank));
    }
    return ptr;
}

inline ze_ipc_mem_handle_t rs_make_ipc_handle_from_fd(ze_ipc_mem_handle_t const& template_handle, int fd) {
    ze_ipc_mem_handle_t handle = template_handle;
    *reinterpret_cast<int*>(&handle) = fd;
    return handle;
}

#if defined(__linux__)
inline void rs_send_ipc_fd(int sock, int fd, int rank, size_t offset) {
    rs_ipc_payload payload{rank, static_cast<uint64_t>(offset)};
    iovec iov{};
    iov.iov_base = &payload;
    iov.iov_len = sizeof(payload);

    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    rs_ipc_exchange_fd cmsg(SOL_SOCKET, SCM_RIGHTS, fd);
    msg.msg_control = &cmsg;
    msg.msg_controllen = sizeof(cmsg);

    if (sendmsg(sock, &msg, 0) == -1) {
        throw std::runtime_error(std::string("sendmsg failed: ") + std::strerror(errno));
    }
}

inline rs_ipc_payload rs_recv_ipc_fd(int sock, int& fd_out) {
    rs_ipc_payload payload{};
    iovec iov{};
    iov.iov_base = &payload;
    iov.iov_len = sizeof(payload);

    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    rs_ipc_exchange_fd cmsg;
    msg.msg_control = &cmsg;
    msg.msg_controllen = sizeof(cmsg);

    if (recvmsg(sock, &msg, 0) == -1) {
        throw std::runtime_error(std::string("recvmsg failed: ") + std::strerror(errno));
    }

    fd_out = cmsg.fd;
    return payload;
}

inline int rs_create_server_socket(std::string const& socket_path) {
    ::unlink(socket_path.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socket_path.c_str());

    int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock == -1) {
        throw std::runtime_error(std::string("socket creation failed: ") + std::strerror(errno));
    }

    auto addr_len = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + std::strlen(addr.sun_path));
    if (::bind(sock, reinterpret_cast<sockaddr*>(&addr), addr_len) == -1) {
        int err = errno;
        ::close(sock);
        throw std::runtime_error(std::string("bind failed: ") + std::strerror(err));
    }

    if (::listen(sock, 2 * 1024) == -1) {
        int err = errno;
        ::close(sock);
        throw std::runtime_error(std::string("listen failed: ") + std::strerror(err));
    }

    return sock;
}

inline int rs_connect_to_server(std::string const& socket_path) {
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socket_path.c_str());

    int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock == -1) {
        throw std::runtime_error(std::string("socket creation failed: ") + std::strerror(errno));
    }

    auto addr_len = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + std::strlen(addr.sun_path));
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), addr_len) == 0) {
            return sock;
        }
        usleep(10000);
    }

    int err = errno;
    ::close(sock);
    throw std::runtime_error(std::string("connect failed: ") + std::strerror(err));
}

inline std::string rs_ipc_session_id(int rank) {
    static std::string session_id;
    if (!session_id.empty()) {
        return session_id;
    }

    char buffer[32] = {};
    if (rank == 0) {
        auto pid = static_cast<unsigned long long>(::getpid());
        auto now = static_cast<unsigned long long>(std::time(nullptr));
        std::snprintf(buffer, sizeof(buffer), "%llx%llx", now, pid);
    }
    MPI_Bcast(buffer, sizeof(buffer), MPI_CHAR, 0, MPI_COMM_WORLD);
    session_id = buffer;
    return session_id;
}

inline std::string rs_make_socket_path(int rank, int exchange_id) {
    return "/tmp/rsipc-" + rs_ipc_session_id(rank) + "-" +
           std::to_string(exchange_id) + "-" + std::to_string(rank);
}
#else
inline void rs_send_ipc_fd(int, int, int, size_t) {
    throw std::runtime_error("Level Zero IPC fd passing requires Linux domain sockets");
}

inline rs_ipc_payload rs_recv_ipc_fd(int, int&) {
    throw std::runtime_error("Level Zero IPC fd passing requires Linux domain sockets");
}

inline int rs_create_server_socket(std::string const&) {
    throw std::runtime_error("Level Zero IPC fd passing requires Linux domain sockets");
}

inline int rs_connect_to_server(std::string const&) {
    throw std::runtime_error("Level Zero IPC fd passing requires Linux domain sockets");
}

inline std::string rs_make_socket_path(int, int) {
    throw std::runtime_error("Level Zero IPC fd passing requires Linux domain sockets");
}
#endif

inline void ze_close_ipc_handle(sycl::context const& ctx, void* ptr) {
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    zeMemCloseIpcHandle(ze_ctx, ptr);
}

// Exchange a device pointer across all MPI ranks via Level Zero IPC.
// Returns an array of world_size device-accessible pointers (caller must free with sycl::free).
// opened_ptrs collects remote pointers that need ze_close_ipc_handle on cleanup.
template <typename T>
T** exchange_ipc_ptrs(T* local_ptr, int rank, int world_size,
                      sycl::queue& Q, std::vector<void*>& opened_ptrs) {
    auto ctx = Q.get_context();
    auto dev = Q.get_device();
    int current_exchange_id = rs_next_exchange_id();

    rs_debug_log(rank, "exchange_ipc_ptrs begin, local_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(local_ptr)));

    auto [local_base, local_offset] = ze_get_ipc_base_and_offset(ctx, local_ptr, rank);
    auto local_handle = ze_get_ipc_handle(ctx, local_base, rank);
    int local_fd = *reinterpret_cast<int*>(&local_handle);

    std::string local_socket = rs_make_socket_path(rank, current_exchange_id);
    int server_sock = rs_create_server_socket(local_socket);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) {
            continue;
        }
        int client_sock = rs_connect_to_server(rs_make_socket_path(peer, current_exchange_id));
        rs_send_ipc_fd(client_sock, local_fd, rank, local_offset);
#if defined(__linux__)
        ::close(client_sock);
#endif
    }

    std::vector<rs_ipc_peer_info> peer_infos(world_size);
    for (int recv_count = 0; recv_count < world_size - 1; ++recv_count) {
#if defined(__linux__)
        int accepted_sock = ::accept(server_sock, nullptr, nullptr);
        if (accepted_sock == -1) {
            int err = errno;
            ::close(server_sock);
            ::unlink(local_socket.c_str());
            throw std::runtime_error(std::string("accept failed: ") + std::strerror(err));
        }
        int remote_fd = -1;
        auto payload = rs_recv_ipc_fd(accepted_sock, remote_fd);
        ::close(accepted_sock);
#else
        int accepted_sock = -1;
        int remote_fd = -1;
        auto payload = rs_recv_ipc_fd(accepted_sock, remote_fd);
#endif
        if (payload.rank < 0 || payload.rank >= world_size) {
            throw std::runtime_error(
                std::string("[rank ") + std::to_string(rank) + "] received invalid IPC rank " +
                std::to_string(payload.rank));
        }
        peer_infos[payload.rank] = {remote_fd, static_cast<size_t>(payload.offset)};
    }

    MPI_Barrier(MPI_COMM_WORLD);

#if defined(__linux__)
    ::close(server_sock);
    ::unlink(local_socket.c_str());
#endif

    rs_debug_log(rank, "exchange_ipc_ptrs fd exchange done");

    T** ptrs = sycl::malloc_shared<T*>(world_size, Q);
    for (int i = 0; i < world_size; i++) {
        if (i == rank) {
            ptrs[i] = local_ptr;
            rs_debug_log(rank, "ipc ptr self set for rank " + std::to_string(i));
        } else {
            if (peer_infos[i].fd < 0) {
                throw std::runtime_error(
                    std::string("[rank ") + std::to_string(rank) + "] missing IPC fd for peer rank " +
                    std::to_string(i));
            }
            auto remote_handle = rs_make_ipc_handle_from_fd(local_handle, peer_infos[i].fd);
            void* remote_base = ze_open_ipc_handle(ctx, dev, remote_handle, rank, i);
            auto* remote = reinterpret_cast<T*>(reinterpret_cast<char*>(remote_base) + peer_infos[i].offset);
            ptrs[i] = remote;
            opened_ptrs.push_back(remote_base);
            rs_debug_log(rank, "ipc ptr opened for peer " + std::to_string(i) +
                                   ", remote_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(remote)) +
                                   ", remote_base=" + std::to_string(reinterpret_cast<uintptr_t>(remote_base)) +
                                   ", offset=" + std::to_string(peer_infos[i].offset));
        }
    }
    rs_debug_log(rank, "exchange_ipc_ptrs end");
    return ptrs;
}

inline void close_ipc_ptrs(sycl::queue& Q, std::vector<void*>& opened_ptrs) {
    auto ctx = Q.get_context();
    for (auto p : opened_ptrs) {
        ze_close_ipc_handle(ctx, p);
    }
    opened_ptrs.clear();
}

template <
    typename TA,
    char LayoutKindA,
    typename TensorA_t,
    typename TC>
class ReduceScatterOp {

public:
    ReduceScatterOp(int m, int n, int k, int rank, int world_size, sycl::queue &Q)
        : m(m), n(n), k(k), rank(rank), world_size(world_size), init_Q(Q) {
    }

    ~ReduceScatterOp() = default;

    // ReduceScatterOp currently reduces the output C across all ranks using MPI.
    //
    // After local GEMM, each rank has its partial C_local = A * B in device memory.
    // This method:
    //   1. Copies C_local from device to host
    //   2. MPI_Allreduce (sum) across all ranks
    //   3. Copies the result back to device
    //
    // After this call, every rank has C_final = Σ C_r.
    void run(TC* device_C, size_t num_elems, sycl::queue& Q) {
        // Allocate host buffers for MPI communication
        std::vector<TC> send_buf(num_elems);
        std::vector<TC> recv_buf(num_elems);

        // Device -> Host
        Q.memcpy(send_buf.data(), device_C, num_elems * sizeof(TC)).wait();

        // MPI AllReduce (sum)
        MPI_Allreduce(send_buf.data(), recv_buf.data(),
                      static_cast<int>(num_elems),
                      mpi_type<TC>(), MPI_SUM, MPI_COMM_WORLD);

        // Host -> Device
        Q.memcpy(device_C, recv_buf.data(), num_elems * sizeof(TC)).wait();
    }

    int get_m() const { return m; }
    int get_n() const { return n; }
    int get_k() const { return k; }
    int get_world_size() const { return world_size; }
    int get_rank() const { return rank; }

private:
    int m, k, n, world_size, rank;
    sycl::queue& init_Q;
};