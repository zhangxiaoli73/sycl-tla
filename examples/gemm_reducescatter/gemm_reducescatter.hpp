#include "reducescatter.hpp"

#include <atomic>
#include <algorithm>
#include <cstring>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;

template <class ATensor, class BTensor, class CTensor,
          class TiledMMA>
void gemm_device(ATensor const& A, // (M,K)
                 BTensor const& B, // (N,K)
                 CTensor& C,       // (M,N)
                 TiledMMA const& mma) {
  // -----
  // Setup
  // -----

  /* Get workgroup and local IDs */
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  /* Create proxy coordinate tensors for each global tensor */
  Tensor cA = make_identity_tensor(A.shape()); // (M,K)
  Tensor cB = make_identity_tensor(B.shape()); // (N,K)
  Tensor cC = make_identity_tensor(C.shape()); // (M,N)

  /* Split GEMM into workgroup tiles, and identify our workgroup's tile
   * (wg_coord) */
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile),
                         make_coord(wg_m, _)); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile),
                         make_coord(wg_n, _)); // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);

  /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  /* Register fragments for MMA */
  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  /* Register fragments for copies */
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  /* Partition global tensor (proxies) for copies */
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition C */
  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
  Tensor tCgC =
      thr_mma.partition_C(gC); /* also matches copy_c's source layout */

  /* Create prefetch TiledCopy instances */
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  /* Partition global tensor (proxies) for prefetch */
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  /* Prefetch distance, in units of k tiles */
  const int prefetch_dist = 3;

  // ------
  // Kernel
  // ------

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  /* Clear the accumulators */
  clear(tCrC);

  /* Warm up loops with prefetch to L1 */
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  /* Main loop */
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    /* Split barrier keeping threads loosely together */
    barrier_arrive(barrier_scope);

    /* Copy A/B from global memory (ideally L1 cache) to registers */
    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    /* Prefetch A/B tiles to L1 */
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

    /* Shuffle data from copy fragments to MMA fragments */
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    /* Accumulate C += A * B */
    gemm(mma, tCrA, tCrB, tCrC);

    /* Other half of split barrier */
    barrier_wait(barrier_scope);
  }

  /* Write C to global memory */
  copy(copy_c, tCrC, tCgC);
}

template <class ATensor, class BTensor, class CTensor,
          class DTensor, class TiledMMA>
void gemm_device_v2(ATensor const& A,       // (M,K)
                    BTensor const& B,       // (N,K)
                    CTensor& C,             // (M,N) local GEMM buffer, IPC-accessible
                    DTensor& D,             // (M,N/world_size) reduce-scatter output
                    TiledMMA const& mma,
                    typename CTensor::element_type** ipc_c_ptrs,  // remote C
                    int** ipc_signal_ptrs,                        // remote flag
                    int rank,
                    int world_size,
                    int m,
                    int n,
                    int num_n_tiles) {
    (void)D;

    // -----
    // Setup
    // -----

    /* Get workgroup and local IDs */
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
    auto wg_m = int(item.get_group(1));
    auto wg_n = int(item.get_group(0));
    auto local_id = int(item.get_local_id(0));

    /* Create proxy coordinate tensors for each global tensor */
    Tensor cA = make_identity_tensor(A.shape()); // (M,K)
    Tensor cB = make_identity_tensor(B.shape()); // (N,K)
    Tensor cC = make_identity_tensor(C.shape()); // (M,N)

    /* Split GEMM into workgroup tiles, and identify our workgroup's tile
     * (wg_coord) */
    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);

    Tensor gA = local_tile(cA, select<0, 2>(wg_tile),
                                                 make_coord(wg_m, _)); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile),
                                                 make_coord(wg_n, _)); // (BLK_N,BLK_K,k)
    Tensor gC =
            local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    /* Create block 2D TiledCopies */
    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);
    auto copy_c = make_block_2d_copy_D(mma, C);

    /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
    auto thr_mma = mma.get_slice(local_id);
    auto thr_copy_a = copy_a.get_slice(local_id);
    auto thr_copy_b = copy_b.get_slice(local_id);

    /* Register fragments for MMA */
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    /* Register fragments for copies */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    /* Partition global tensor (proxies) for copies */
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    /* Partition C and D */
    Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
    Tensor tCgC =
            thr_mma.partition_C(gC); /* also matches copy_c's source layout */

    /* Create prefetch TiledCopy instances */
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);

    auto thr_prefetch_A = prefetch_a.get_slice(local_id);
    auto thr_prefetch_B = prefetch_b.get_slice(local_id);

    /* Partition global tensor (proxies) for prefetch */
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    /* Prefetch distance, in units of k tiles */
    const int prefetch_dist = 3;

    // ------
    // Kernel
    // ------

    constexpr int barrier_scope = 2;

    int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
    int k_tile_prefetch = 0;

    /* Clear the accumulators */
    clear(tCrC);

    /* Warm up loops with prefetch to L1 */
    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
        prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    /* Main loop */
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
        /* Split barrier keeping threads loosely together */
        barrier_arrive(barrier_scope);

        /* Copy A/B from global memory (ideally L1 cache) to registers */
        copy(copy_a, tAgA(_, _, _, k_tile), tArA);
        copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

        /* Prefetch A/B tiles to L1 */
        prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

        /* Shuffle data from copy fragments to MMA fragments */
        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);

        /* Accumulate C += A * B */
        gemm(mma, tCrA, tCrB, tCrC);

        /* Other half of split barrier */
        barrier_wait(barrier_scope);
    }

    /* Materialize the local GEMM tile before exposing it through IPC. */
    copy(copy_c, tCrC, tCgC);
    item.barrier(sycl::access::fence_space::global_and_local);

    auto* local_c_ptr = C.data().get();
    int tile_m = int(get<0>(wg_tile));
    int tile_n = int(get<1>(wg_tile));
    int tile_row_start = wg_m * tile_m;
    int tile_col_start = wg_n * tile_n;
    int cols_in_tile = std::max(0, std::min(tile_n, n - tile_col_start));

        if (cols_in_tile > 0 && tile_row_start < m) {
                int tile_id = wg_m * num_n_tiles + wg_n;
                if (local_id == 0) {
                        auto* signal_base = ipc_signal_ptrs[rank];
                        signal_base[tile_id * world_size + rank] = 1;
                }
    }

}

template <class CTensor, class DTensor, class TiledMMA>
void reduce_scatter_consumer_device(
        CTensor& C,                           // (M,N) local GEMM buffer
        DTensor& D,                           // (M,N/world_size) local reduce-scatter output
        TiledMMA const& mma,
        typename CTensor::element_type** ipc_c_ptrs,
        int** ipc_signal_ptrs,
        int rank,
        int world_size,
        int m,
        int n,
        int num_n_tiles) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
    int wg_m = int(item.get_group(1));
    int wg_n = int(item.get_group(0));
    int local_id = int(item.get_local_id(0));
    int local_threads = int(item.get_local_range(0));

    int tile_m = int(get<0>(mma.tile_mnk()));
    int tile_n = int(get<1>(mma.tile_mnk()));

    int row_start = wg_m * tile_m;
    int row_end = std::min(row_start + tile_m, m);
    int rows = row_end - row_start;
    if (rows <= 0) {
        return;
    }

    int col_start = wg_n * tile_n;
    int col_end = std::min(col_start + tile_n, n);
    int n_per_rank = n / world_size;
    int rank_col_start = rank * n_per_rank;
    int rank_col_end = rank_col_start + n_per_rank;
    int owned_col_start = std::max(col_start, rank_col_start);
    int owned_col_end = std::min(col_end, rank_col_end);
    int cols = owned_col_end - owned_col_start;
    if (cols <= 0) {
        return;
    }

    int tile_id = wg_m * num_n_tiles + wg_n;

    // Wait until every rank has produced this tile into its local C buffer.
    for (int src_rank = 0; src_rank < world_size; ++src_rank) {
        auto* signal_base = ipc_signal_ptrs[src_rank];
        while (signal_base[tile_id * world_size + src_rank] == 0) {
        }
    }

    auto* d_ptr = D.data().get();
    int d_ld = n_per_rank;

    for (int linear_idx = local_id; linear_idx < rows * cols; linear_idx += local_threads) {
        int row = linear_idx / cols;
        int col = linear_idx % cols;
        int global_row = row_start + row;
        int global_col = owned_col_start + col;

        float acc = 0.0f;
        for (int src_rank = 0; src_rank < world_size; ++src_rank) {
            auto* src_c_ptr = ipc_c_ptrs[src_rank];
            acc += static_cast<float>(src_c_ptr[global_row * n + global_col]);
        }

        int d_col = global_col - rank_col_start;
        d_ptr[global_row * d_ld + d_col] = static_cast<typename DTensor::element_type>(acc);
    }
}

template <typename TA, typename TB, typename TC>
auto choose_mma_op() {
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>)
    return XE_DPAS_TT<8, TC, TA, TB>{};
  else if constexpr (is_same_v<TA, cute::bfloat16_t>)
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  else /* Use f16 by default as upconversion sequences are typically faster */
    return XE_DPAS_TT<8, float, cute::half_t>{};
}

template <class ATensor, class BTensor, class CTensor>
auto choose_tiled_mma(ATensor const& A, BTensor const& B, CTensor const&) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op<TA, TB, TC>();

  constexpr bool byte = (cute::max(sizeof_bits_v<TA>, sizeof_bits_v<TB>) <= 8);
  constexpr bool a_t = is_constant_v<1, decltype(stride<0>(A))>;
  constexpr bool b_n = is_constant_v<1, decltype(stride<0>(B))>;

  constexpr bool use_1x_dpas_per_k =
      a_t               // Use one DPAS in k dimension for A^T case
      || (byte && b_n); //  pending compiler improvements (also int8 B^N).
  constexpr bool use_4x8_sg =
      ((sizeof_bits_v<TB> <
        sizeof_bits_v<TA>) // Use smaller B loads for expensive reorders.
       &&!(is_same_v<TB, cute::float_e5m2_t>) ) ||
      (b_n && sizeof_bits_v<TB> < 8);

  using _K = conditional_t<use_1x_dpas_per_k, C<op.K>, C<op.K * 2>>;

  using WGTile = Shape<_256, _256, _K>; // 256x256 WG tile size
  using SGLayout8x4 =
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>; // 8x4 SG tiling, n-major
  using SGLayout4x8 =
      Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>; // 4x8 SG tiling, n-major
  using SGLayout = conditional_t<use_4x8_sg, SGLayout4x8, SGLayout8x4>;

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;

  return MMA{};
}

template <class, class, char, char, int>
class GemmCuteName;

template <class, class, char, char, int>
class GemmCuteFusedName;

template <class, class, char, char, int>
class ReduceScatterConsumerName;


// template <class EngineA, class LayoutA, class TensorB_t, class MaxTensorC_t, class TA, class TB, char LayoutA, char LayoutB>
template <class TA, class TB, class TC, char LayoutKindA, char LayoutKindB>
class GemmAllReduce {
    using TensorA_t = Tensor<ViewEngine<gmem_ptr<TA*>>, Layout<tuple<int, int>, tuple<int, C<1>>>>;
    using TensorB_t = Tensor<ViewEngine<gmem_ptr<TB*>>, Layout<tuple<int, int>, tuple<C<1>, int>>>;
    using TensorC_t = Tensor<ViewEngine<gmem_ptr<TC*>>, Layout<tuple<int, int>, tuple<int, C<1>>>>;
    // D: reduce-scatter output, shape (M, N/world_size), same element type as C.
    using TensorD_t = Tensor<ViewEngine<gmem_ptr<TC*>>, Layout<tuple<int, int>, tuple<int, C<1>>>>;

    public:
    GemmAllReduce(int m, int n, int k, int rank, int world_size,
                   TensorA_t const& A, TensorB_t const& B, TensorC_t& C, TensorD_t& D,
                   sycl::queue& Q, bool fusion_enabled = true)
                : ar_op(m, n, k, rank, world_size, Q),
                    gemm_q(Q),
                    rs_q(Q.get_context(), Q.get_device()),
                    fusion_enabled_(fusion_enabled) {
        if (fusion_enabled_) {
            auto mma = choose_tiled_mma(A, B, C);
            initialize_fused_ipc(C, mma, Q);
        }
    }
    ~GemmAllReduce() {
        release_fused_ipc();
    }

    // Launch the local GEMM kernel: C = A * B
    template <class MMAType>
    void gemm_cute(sycl::queue& Q,
                TensorA_t const& A, // (M, K)
                TensorB_t const& B, // (N, K)
                TensorC_t& C,       // (M, N)
                MMAType mma) {
        namespace syclex = sycl::ext::oneapi::experimental;
        namespace intelex = sycl::ext::intel::experimental;

        syclex::properties kernel_props{syclex::sub_group_size<16>,
                                        intelex::grf_size<256>};

        int m = ar_op.get_m();

        sycl::range<2> local = {size(mma), 1};
        sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(m, get<0>(mma.tile_mnk()))};
        Q.submit([&](sycl::handler &h) {
            h.parallel_for<GemmCuteName<TA, TB, LayoutKindA, LayoutKindB, 0>>(
                sycl::nd_range<2>(global, local), kernel_props,
                [=](auto) {
                    gemm_device(A, B, C, mma);
                });
        });
    }

    // Launch GEMM kernel with in-kernel reduce-scatter fusion.
    template <class MMAType>
    void gemm_cute_fused(sycl::queue& Q,
                TensorA_t const& A,
                TensorB_t const& B,
                TensorC_t& C,
                TensorD_t& D,
                MMAType mma,
                TC** ipc_c_ptrs,
                int** ipc_signal_ptrs,
                int rank,
                int world_size,
                int m,
                int n,
                int num_n_tiles) {
        namespace syclex = sycl::ext::oneapi::experimental;
        namespace intelex = sycl::ext::intel::experimental;

        syclex::properties kernel_props{syclex::sub_group_size<16>,
                                        intelex::grf_size<256>};

        sycl::range<2> local = {size(mma), 1};
        sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(m, get<0>(mma.tile_mnk()))};
        Q.submit([&](sycl::handler &h) {
            h.parallel_for<GemmCuteFusedName<TA, TB, LayoutKindA, LayoutKindB, 0>>(
                sycl::nd_range<2>(global, local), kernel_props,
                [=](sycl::nd_item<2>) {
                    gemm_device_v2(A, B, C, D, mma,
                                   ipc_c_ptrs, ipc_signal_ptrs,
                                   rank, world_size, m, n, num_n_tiles);
                });
        });
    }

    template <class MMAType>
    void reduce_scatter_consumer(sycl::queue& Q,
                TensorC_t& C,
                TensorD_t& D,
                MMAType mma,
                TC** ipc_c_ptrs,
                int** ipc_signal_ptrs,
                int rank,
                int world_size,
                int m,
                int n,
                int num_n_tiles) {
        namespace syclex = sycl::ext::oneapi::experimental;
        namespace intelex = sycl::ext::intel::experimental;

        syclex::properties kernel_props{syclex::sub_group_size<16>,
                                        intelex::grf_size<256>};

        sycl::range<2> local = {size(mma), 1};
        sycl::range<2> global = {local[0] * num_n_tiles,
                                 local[1] * ceil_div(m, get<0>(mma.tile_mnk()))};
        Q.submit([&](sycl::handler &h) {
            h.parallel_for<ReduceScatterConsumerName<TA, TB, LayoutKindA, LayoutKindB, 0>>(
                sycl::nd_range<2>(global, local), kernel_props,
                [=](sycl::nd_item<2>) {
                    reduce_scatter_consumer_device(
                        C, D, mma, ipc_c_ptrs, ipc_signal_ptrs,
                        rank, world_size, m, n, num_n_tiles);
                });
        });
    }

    bool is_fusion_enabled() const { return fusion_enabled_; }

    template <class MMAType>
    void initialize_fused_ipc(TensorC_t& C, MMAType mma, sycl::queue& Q) {
        TC* c_ptr = C.data().get();

        int world_size = ar_op.get_world_size();
        int rank = ar_op.get_rank();
        int m = ar_op.get_m();
        int n = ar_op.get_n();
        int tile_m = int(get<0>(mma.tile_mnk()));
        int tile_n = int(get<1>(mma.tile_mnk()));
        int num_m_tiles = int(ceil_div(m, tile_m));
        int num_n_tiles = int(ceil_div(n, tile_n));
        int required_tiles = num_m_tiles * num_n_tiles;

        // If C pointer changed or tile shape changed, rebuild IPC resources.
        if (fused_ipc_initialized_ && (local_c_ptr_ != c_ptr || num_tiles_ != required_tiles)) {
            rs_debug_log(rank, "initialize_fused_ipc detected shape/pointer change, releasing old IPC resources");
            release_fused_ipc();
        }

        if (fused_ipc_initialized_) {
            return;
        }

        local_c_ptr_ = c_ptr;
        num_tiles_ = required_tiles;

        rs_debug_log(rank, "initialize_fused_ipc allocate buffers, num_tiles=" + std::to_string(num_tiles_) +
                           ", world_size=" + std::to_string(world_size));

        signal_local_ = sycl::malloc_device<int>(num_tiles_ * world_size, Q);
        // Signal buffer is initialized once here and then reused.
        Q.memset(signal_local_, 0, sizeof(int) * num_tiles_ * world_size).wait();

        rs_debug_log(rank, "initialize_fused_ipc local buffers ready");

        ipc_c_ptrs_ = exchange_ipc_ptrs(local_c_ptr_, rank, world_size, Q, opened_c_ptrs_);
        ipc_signal_ptrs_ = exchange_ipc_ptrs(signal_local_, rank, world_size, Q, opened_signal_ptrs_);

        rs_debug_log(rank, "initialize_fused_ipc remote IPC pointers ready");

        fused_ipc_initialized_ = true;
        rs_debug_log(rank, "initialize_fused_ipc complete");
    }

    void release_fused_ipc() {
        if (!fused_ipc_initialized_) {
            return;
        }

        close_ipc_ptrs(gemm_q, opened_c_ptrs_);
        close_ipc_ptrs(gemm_q, opened_signal_ptrs_);

        if (ipc_c_ptrs_) sycl::free(ipc_c_ptrs_, gemm_q);
        if (ipc_signal_ptrs_) sycl::free(ipc_signal_ptrs_, gemm_q);
        if (signal_local_) sycl::free(signal_local_, gemm_q);

        ipc_c_ptrs_ = nullptr;
        ipc_signal_ptrs_ = nullptr;
        signal_local_ = nullptr;
        local_c_ptr_ = nullptr;
        num_tiles_ = 0;
        fused_ipc_initialized_ = false;
    }

    // GEMM + AllReduce pattern:
    //   1. Each rank computes C_local = A * B (local GEMM)
    //   2. MPI_Allreduce sums all C_locals: C_final = Σ C_r
    //   After this, every rank has the same C_final.
    void run(
            TensorA_t const& A,
            TensorB_t const& B,
            TensorC_t& C,
            TensorD_t& D,
            sycl::queue& Q) {
        if (fusion_enabled_) {
            run_fused(A, B, C, D, Q);
        } else {
            run_gemm(A, B, C, Q);
        }
    }

    // Fused path: tile-level GEMM + reduce-scatter in one kernel.
    // Tile ownership is assigned by tile_id % world_size.
    // C (m×n): local GEMM buffer exposed via IPC.
    // D (m×n/world_size): reduce-scatter output for this rank.
    void run_fused(
            TensorA_t const& A,
            TensorB_t const& B,
            TensorC_t& C,
            TensorD_t& D,
            sycl::queue& Q) {
        auto mma = choose_tiled_mma(A, B, C);

        int rank = ar_op.get_rank();
        int world_size = ar_op.get_world_size();
        int m = ar_op.get_m();
        int n = ar_op.get_n();
        int tile_m = int(get<0>(mma.tile_mnk()));
        int tile_n = int(get<1>(mma.tile_mnk()));
        int num_n_tiles = int(ceil_div(n, tile_n));
        assert(fused_ipc_initialized_ && "fused IPC must be initialized in constructor");

        gemm_q.memset(signal_local_, 0, sizeof(int) * num_tiles_ * world_size).wait();

        rs_debug_log(rank, "run_fused launching fused kernel");

        // Launch decoupled consumer first so it can wait for ready tiles while GEMM produces them.
        reduce_scatter_consumer<decltype(mma)>(
            rs_q, C, D, mma,
            ipc_c_ptrs_, ipc_signal_ptrs_, rank, world_size,
            m, n, num_n_tiles);

        // Launch GEMM producer kernel.
        gemm_cute_fused<decltype(mma)>(
            gemm_q, A, B, C, D, mma,
            ipc_c_ptrs_, ipc_signal_ptrs_, rank, world_size,
            m, n, num_n_tiles);

        rs_debug_log(rank, "run_fused kernel submitted, waiting for completion");
        gemm_q.wait_and_throw();
        rs_q.wait_and_throw();
        rs_debug_log(rank, "run_fused complete");
    }

    // Separate path: GEMM then AllReduce as two independent steps
    void run_gemm(
            TensorA_t const& A,
            TensorB_t const& B,
            TensorC_t& C,
            sycl::queue& Q) {
        // Step 1: local GEMM
        auto mma = choose_tiled_mma(A, B, C);
        gemm_cute<decltype(mma)>(gemm_q, A, B, C, mma);
        gemm_q.wait_and_throw();

        // Step 2: AllReduce via MPI (no overlap with GEMM)
        // todo: disable
        // size_t num_elems = static_cast<size_t>(ar_op.get_m()) * ar_op.get_n();
        // ar_op.run(C.data().get(), num_elems, Q);
    }

private:
    ReduceScatterOp<TA, LayoutKindA, TensorA_t, TC> ar_op;
    sycl::queue gemm_q;
    sycl::queue rs_q;
    bool fusion_enabled_ = true;

    // Fused-mode IPC resources (initialized once, reused across runs).
    bool fused_ipc_initialized_ = false;
    int num_tiles_ = 0;
    TC* local_c_ptr_ = nullptr;
    int* signal_local_ = nullptr;
    TC** ipc_c_ptrs_ = nullptr;
    int** ipc_signal_ptrs_ = nullptr;
    std::vector<void*> opened_c_ptrs_;
    std::vector<void*> opened_signal_ptrs_;
};