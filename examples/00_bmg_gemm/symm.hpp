#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include <level_zero/ze_api.h>

#if defined(__linux__)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

// MPI datatype helper
template <typename T>
MPI_Datatype mpi_type();

template <>
inline MPI_Datatype mpi_type<float>() {
  return MPI_FLOAT;
}

template <>
inline MPI_Datatype mpi_type<double>() {
  return MPI_DOUBLE;
}

// ---------------- Level Zero IPC helpers ----------------

inline ze_ipc_mem_handle_t ze_get_ipc_handle(sycl::context const& ctx, void* ptr) {
  if (ptr == nullptr) {
    throw std::runtime_error("ze_get_ipc_handle received null pointer");
  }
  auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  ze_ipc_mem_handle_t handle = {};
  auto res = zeMemGetIpcHandle(ze_ctx, ptr, &handle);
  if (res != ZE_RESULT_SUCCESS) {
    throw std::runtime_error("zeMemGetIpcHandle failed");
  }
  return handle;
}

inline std::pair<void*, size_t> ze_get_ipc_base_and_offset(
    sycl::context const& ctx,
    void* ptr) {
  auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  void* base_addr = nullptr;
  size_t base_size = 0;
  auto res = zeMemGetAddressRange(ze_ctx, ptr, &base_addr, &base_size);
  if (res != ZE_RESULT_SUCCESS || base_addr == nullptr || base_size == 0) {
    throw std::runtime_error("zeMemGetAddressRange failed or returned invalid range");
  }
  auto offset = static_cast<size_t>(reinterpret_cast<char*>(ptr) - reinterpret_cast<char*>(base_addr));
  return {base_addr, offset};
}

inline void* ze_open_ipc_handle(sycl::context const& ctx, sycl::device const& dev, ze_ipc_mem_handle_t handle) {
  auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
  void* ptr = nullptr;
  auto res = zeMemOpenIpcHandle(ze_ctx, ze_dev, handle, 0, &ptr);
  if (res != ZE_RESULT_SUCCESS || ptr == nullptr) {
    throw std::runtime_error("zeMemOpenIpcHandle failed or returned null pointer");
  }
  return ptr;
}

inline ze_ipc_mem_handle_t rs_make_ipc_handle_from_fd(ze_ipc_mem_handle_t const& template_handle, int fd) {
  ze_ipc_mem_handle_t handle = template_handle;
  *reinterpret_cast<int*>(&handle) = fd;
  return handle;
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

  if (::listen(sock, 2048) == -1) {
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
    std::snprintf(buffer, sizeof(buffer), "%llx", pid);
  }
  MPI_Bcast(buffer, sizeof(buffer), MPI_CHAR, 0, MPI_COMM_WORLD);
  session_id = buffer;
  return session_id;
}

inline std::string rs_make_socket_path(int rank, int exchange_id) {
  return "/tmp/symmipc-" + rs_ipc_session_id(rank) + "-" +
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
  if (ptr == nullptr) {
    return;
  }
  auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  auto res = zeMemCloseIpcHandle(ze_ctx, ptr);
  if (res != ZE_RESULT_SUCCESS) {
    throw std::runtime_error("zeMemCloseIpcHandle failed");
  }
}

template <typename T>
T** exchange_ipc_ptrs(T* local_ptr, int rank, int world_size, sycl::queue& q, std::vector<void*>& opened_ptrs) {
  auto ctx = q.get_context();
  auto dev = q.get_device();
  int current_exchange_id = rs_next_exchange_id();

  auto [local_base, local_offset] = ze_get_ipc_base_and_offset(ctx, local_ptr);
  auto local_handle = ze_get_ipc_handle(ctx, local_base);
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
      throw std::runtime_error("received invalid IPC rank");
    }
    peer_infos[payload.rank] = {remote_fd, static_cast<size_t>(payload.offset)};
  }

  MPI_Barrier(MPI_COMM_WORLD);

#if defined(__linux__)
  ::close(server_sock);
  ::unlink(local_socket.c_str());
#endif

  T** ptrs = new T*[world_size];
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) {
      ptrs[peer] = local_ptr;
    } else {
      if (peer_infos[peer].fd < 0) {
        throw std::runtime_error("exchange_ipc_ptrs missing IPC fd for peer");
      }
      auto remote_handle = rs_make_ipc_handle_from_fd(local_handle, peer_infos[peer].fd);
      void* remote_base = ze_open_ipc_handle(ctx, dev, remote_handle);
      auto* remote = reinterpret_cast<T*>(reinterpret_cast<char*>(remote_base) + peer_infos[peer].offset);
      if (remote == nullptr) {
        throw std::runtime_error("exchange_ipc_ptrs got null remote pointer");
      }
      ptrs[peer] = remote;
      opened_ptrs.push_back(remote_base);
    }
  }
  return ptrs;
}

inline void close_ipc_ptrs(sycl::queue& q, std::vector<void*>& opened_ptrs) {
  auto ctx = q.get_context();
  for (void* p : opened_ptrs) {
    ze_close_ipc_handle(ctx, p);
  }
  opened_ptrs.clear();
}

// ---------------- Symmetric-memory style signal barrier ----------------

// This class provides barrier/put_signal/wait_signal semantics similar to
// XPUSymmetricMemory. Each rank owns one signal pad, IPC-opens all peer pads,
// and stores per-channel tickets in layout: [channel][src_rank].
class SymmMemory {
 public:
  SymmMemory(int m, int n, int k, int rank, int world_size, sycl::queue& q, int num_channels = 1024)
      : m_(m),
        n_(n),
        k_(k),
        rank_(rank),
        world_size_(world_size),
        num_channels_(num_channels),
        init_q_(q),
        local_epoch_(num_channels, 0) {
      // Data buffer layout for allgathered A shards: [world_size][m][k].
      // `m` here is local_m from caller, so total elements are world_size * local_m * k.
    size_t data_elems = static_cast<size_t>(m) * k * world_size_; // 16-bit elements
    size_t signal_elems = static_cast<size_t>(num_channels_) * world_size_;

    std::cout << "zl_debug start to malloc local buffer and flag " << std::endl;
    local_signal_ptr_ = sycl::malloc_device<uint32_t>(signal_elems, init_q_);
    local_data_ptr_ = sycl::malloc_device<uint16_t>(data_elems, init_q_);
    std::cout << "zl_debug finish malloc local buffer and flag " << std::endl;

    init_q_.memset(local_signal_ptr_, 0, signal_elems * sizeof(uint32_t)).wait();
    init_q_.memset(local_data_ptr_, 0, data_elems * sizeof(uint16_t)).wait();
    
    std::cout << "zl_debug start to do IPC exchange " << std::endl;
    remote_signal_ptrs_ = exchange_ipc_ptrs(local_signal_ptr_, rank_, world_size_, init_q_, opened_signal_bases_);
    remote_data_ptrs_ = exchange_ipc_ptrs(local_data_ptr_, rank_, world_size_, init_q_, opened_data_bases_);
    std::cout << "zl_debug finish the IPC exchange " << std::endl;

    // make remote IPC memory resident on local device
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(init_q_.get_context());
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(init_q_.get_device());
    
    std::cout << "zl_debug start to make resident" << std::endl;
    for (int peer = 0; peer < world_size_; ++peer) {
      if (peer == rank_) continue;
      if (remote_signal_ptrs_[peer] == nullptr) {
        throw std::runtime_error("SymmMemory remote_signal_ptr is null for peer " + std::to_string(peer));
      }
      auto res = zeContextMakeMemoryResident(ze_ctx, ze_dev, remote_signal_ptrs_[peer],
          signal_elems * sizeof(uint32_t));
      if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("zeContextMakeMemoryResident failed for remote signal ptr of peer " + std::to_string(peer));
      }
    }

    for (int peer = 0; peer < world_size_; ++peer) {
      if (peer == rank_) continue;
      if (remote_data_ptrs_[peer] == nullptr) {
        throw std::runtime_error("SymmMemory remote_data_ptr is null for peer " + std::to_string(peer));
      }
      auto res = zeContextMakeMemoryResident(ze_ctx, ze_dev, remote_data_ptrs_[peer],
          data_elems * sizeof(uint16_t));
      if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("zeContextMakeMemoryResident failed for remote data ptr of peer " + std::to_string(peer));
      }
    }
    std::cout << "zl_debug finish making resident" << std::endl;
  }

  ~SymmMemory() {
    close_ipc_ptrs(init_q_, opened_signal_bases_);
    close_ipc_ptrs(init_q_, opened_data_bases_);

    if (remote_signal_ptrs_) {
      delete[] remote_signal_ptrs_;
      remote_signal_ptrs_ = nullptr;
    }
    delete[] remote_data_ptrs_;
    remote_data_ptrs_ = nullptr;
    if (local_signal_ptr_) {
      sycl::free(local_signal_ptr_, init_q_);
      local_signal_ptr_ = nullptr;
    }
    if (local_data_ptr_) {
      sycl::free(local_data_ptr_, init_q_);
      local_data_ptr_ = nullptr;
    }
  }

  SymmMemory(SymmMemory const&) = delete;
  SymmMemory& operator=(SymmMemory const&) = delete;

  int get_m() const { return m_; }
  int get_n() const { return n_; }
  int get_k() const { return k_; }
  int get_rank() const { return rank_; }
  int get_world_size() const { return world_size_; }

  void check_channel(int channel) const {
    assert(channel >= 0 && "channel must be non-negative");
    assert(channel < num_channels_ && "channel exceeds configured num_channels");
  }

  // Equivalent semantics to XPUSymmetricMemory::barrier(channel):
  // - publish a ticket to peers' signal pads
  // - wait until every peer publishes the same ticket to this rank
  void barrier(int channel, size_t timeout_ms = 0) {
    barrier(channel, init_q_, timeout_ms);
  }

  void barrier(int channel, sycl::queue& queue, size_t timeout_ms = 0) {
    (void)timeout_ms;
    (void)channel;

    constexpr int barrier_channel = 0;

    uint32_t ticket = ++local_epoch_[barrier_channel];
    uint32_t** pads = remote_signal_ptrs_;
    int rank = rank_;
    int world_size = world_size_;
    int base = barrier_channel * world_size_;
    std::cout << "zl_debug start to do memset " << std::endl;
    // memset to 0 before the first barrier
     for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) {
        continue;
      }
      uint32_t* remote_slot = reinterpret_cast<uint32_t*>(remote_signal_ptrs_[peer]);
      if (remote_slot) {
        throw std::runtime_error("SymmMemory barrier remote_slot is null, unexpected.");
      }
      queue.memset(remote_slot, 0, world_size_ * sizeof(uint32_t)).wait();
    }

    std::cout << "zl_debug memset done" << std::endl;
    return;

    queue.submit([&](sycl::handler& h) {
      h.single_task([=]() {
        // put_signal to all peers
        for (int peer = 0; peer < world_size; ++peer) {
          if (peer == rank) {
            continue;
          }
          uint32_t* remote_slot = pads[peer] + base + rank;
          remote_slot[0] = 0;
          sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        }

        // wait_signal from all peers
        // uint32_t* my_pad = pads[rank] + base;
        // for (int peer = 0; peer < world_size; ++peer) {
        //   if (peer == rank) {
        //     continue;
        //   }
        //   while (true) {
        //     sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        //     uint32_t* wait_slot = my_pad + peer;
        //     if (*wait_slot >= ticket) {
        //       break;
        //     }
        //   }
        // }
      });
    });
  }

  void put_signal(int dst_rank, int channel, size_t timeout_ms = 0) {
    (void)timeout_ms;
    check_channel(channel);
    assert(dst_rank >= 0 && dst_rank < world_size_);

    uint32_t ticket = ++local_epoch_[channel];
    int rank = rank_;
    int base = channel * world_size_;
    uint32_t* remote_pad = remote_signal_ptrs_[dst_rank];

    init_q_.submit([&](sycl::handler& h) {
      h.single_task([=]() {
        uint32_t* remote_slot = remote_pad + base + rank;
        *remote_slot = ticket;
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
      });
    });
  }

  void wait_signal(int src_rank, int channel, size_t timeout_ms = 0) {
    (void)timeout_ms;
    check_channel(channel);
    assert(src_rank >= 0 && src_rank < world_size_);

    uint32_t ticket = local_epoch_[channel];
    int rank = rank_;
    int base = channel * world_size_;
    uint32_t* my_pad = remote_signal_ptrs_[rank];

    init_q_.submit([&](sycl::handler& h) {
      h.single_task([=]() {
        uint32_t* wait_slot = my_pad + base + src_rank;
        while (true) {
          sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
          if (*wait_slot >= ticket) {
            break;
          }
        }
      });
    });
  }

  uint32_t* local_signal_ptr() const { return local_signal_ptr_; }
  uint32_t** remote_signal_ptrs() const { return remote_signal_ptrs_; }
  uint16_t* local_data_ptr() const { return local_data_ptr_; }
  uint16_t** remote_data_ptrs() const { return remote_data_ptrs_; }

 private:
  int m_;
  int n_;
  int k_;
  int rank_;
  int world_size_;
  int num_channels_;

  sycl::queue& init_q_;

  uint32_t* local_signal_ptr_ = nullptr;
  uint32_t** remote_signal_ptrs_ = nullptr;
  uint16_t* local_data_ptr_ = nullptr;
  uint16_t** remote_data_ptrs_ = nullptr;
  std::vector<void*> opened_signal_bases_;
  std::vector<void*> opened_data_bases_;
  std::vector<uint32_t> local_epoch_;
};
