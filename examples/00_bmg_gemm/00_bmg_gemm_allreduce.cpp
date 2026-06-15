
#include <mpi.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cute/tensor.hpp>

#include "symm.hpp"
#include "gemm_allreduce_kernel.hpp"
#include "cutlass/util/command_line.h"
#include "helper.h"
#include "sycl_common.hpp"

using namespace cute;

struct Options {
	bool help = false;
	bool error = false;
	int m = 256; //8192;
	int n = 2048; // hidden size
	int k = 1536; // moe intermediate size
	int iterations = 20;
	int debug_log = 1;
	int verify = 0;
	int use_push_mode = 1;
	float alpha = 1.0f;

	void parse(int argc, char **args) {
		std::vector<char const*> cargs(argc);
		for (int i = 0; i < argc; ++i) {
			cargs[i] = args[i];
		}
		cutlass::CommandLine cmd(argc, cargs.data());
		if (cmd.check_cmd_line_flag("help")) {
			help = true;
			return;
		}
		cmd.get_cmd_line_argument("m", m, 8192);
		cmd.get_cmd_line_argument("n", n, 4096);
		cmd.get_cmd_line_argument("k", k, 3584);
		cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
		cmd.get_cmd_line_argument("iterations", iterations, 20);
		cmd.get_cmd_line_argument("debug_log", debug_log, 1);
		cmd.get_cmd_line_argument("verify", verify, 0);
		cmd.get_cmd_line_argument("use_push_mode", use_push_mode, 1);
	}

	std::ostream& print_usage(std::ostream& out) const {
		out << "BMG GEMM Allreduce Example (Option A: fused cute kernel)\n\n"
			<< "  --m=<int>         M extent\n"
			<< "  --n=<int>         N extent\n"
			<< "  --k=<int>         K extent\n"
			<< "  --alpha=<float>   alpha scaling (default 1.0)\n"
			<< "  --iterations=<int>\n"
			<< "  --debug_log=<int> 0/1 progress logs (default 1)\n"
			<< "  --verify=<int>    0/1 run accuracy verification (default 0)\n"
			<< "  --use_push_mode=<int> 1=push, 0=pull (default 1)\n\n";
		return out;
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Tensor type aliases (RowMajor A, RowMajor B, RowMajor C/D)
/////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementC = bfloat16_t;  // IPC-shared buffer and output

using TensorA_t = Tensor<ViewEngine<gmem_ptr<ElementA*>>,
                         Layout<tuple<int, int>, tuple<int, C<1>>>>;  // (M,K) RowMajor
using TensorB_t = Tensor<ViewEngine<gmem_ptr<ElementB*>>,
                         Layout<tuple<int, int>, tuple<C<1>, int>>>;  // (N,K) ColMajor-ish (RowMajor B with stride(ldb, 1) → actually (N,K) with stride(1, ldb))
using TensorC_t = Tensor<ViewEngine<gmem_ptr<ElementC*>>,
                         Layout<tuple<int, int>, tuple<int, C<1>>>>;  // (M,N) RowMajor
using TensorD_t = TensorC_t;  // same layout

/////////////////////////////////////////////////////////////////////////////////////////////////
// Runner class: manages IPC, launches fused kernel, benchmarks
/////////////////////////////////////////////////////////////////////////////////////////////////

struct ExampleRunner {
	std::unique_ptr<sycl::queue> q_;
	std::unique_ptr<SymmMemory> symm_;

	// Fused kernel IPC signal resources (tile-level)
	int* signal_local_ = nullptr;
	int num_tiles_ = 0;

	// IPC output pointers for RS+AG allgather phase
	std::vector<void*> opened_out_bases_;
	void** ipc_out_ptrs_dev_ = nullptr;

	void initialize_tile_signals(int m, int n, int k, int rank, int world_size,
	                             TensorA_t const& A, TensorB_t const& B, TensorC_t const& C_meta) {
		auto mma = choose_tiled_mma_ar(A, B, C_meta);
		int tile_m = int(get<0>(mma.tile_mnk()));
		int tile_n = int(get<1>(mma.tile_mnk()));
		int num_m_tiles = int(ceil_div(m, tile_m));
		int num_n_tiles = int(ceil_div(n, tile_n));
		num_tiles_ = num_m_tiles * num_n_tiles;
		size_t signal_elems = static_cast<size_t>(num_tiles_) * world_size;
		// Ensure signal buffer is large enough for RS+AG per-WG barrier slots
		size_t rs_barrier_size = static_cast<size_t>(kRSBarrierBaseU32)
		    + static_cast<size_t>(kRSAG_MaxReducerWGs) * world_size;
		signal_elems = std::max(signal_elems, rs_barrier_size);

		if (!symm_) {
			size_t override_data_elems = static_cast<size_t>(m) * n * world_size;
			size_t override_signal_elems = signal_elems;
			symm_ = std::make_unique<SymmMemory>(
			    m, n, k, rank, world_size, *q_, 8,
			    override_data_elems, override_signal_elems);
		}

		signal_local_ = reinterpret_cast<int*>(symm_->local_signal_ptr_);
		q_->memset(signal_local_, 0, sizeof(int) * signal_elems).wait();
	}

	void release_tile_signals() {
		signal_local_ = nullptr;
		num_tiles_ = 0;
	}

	// Launch fused GEMM + Allreduce kernel using precomputed launch config
	template <class MMAType>
	void run_fused(
			TensorA_t const& A,
			TensorB_t const& B,
			TensorC_t& C,
			TensorD_t& D,
			MMAType const& mma,
			sycl::range<2> const& local,
			sycl::range<2> const& global,
			int num_n_tiles,
			void** ipc_data_ptrs,
			int** ipc_signal_ptrs,
			void** ipc_out_ptrs,
			float alpha,
			uint32_t signal_token,
			bool use_push_mode,
			int rank,
			int world_size,
			int m,
			int n,
			int num_reducer_wgs,
			bool debug_log = false) {
		namespace syclex = sycl::ext::oneapi::experimental;
		namespace intelex = sycl::ext::intel::experimental;

		syclex::properties kernel_props{syclex::sub_group_size<16>,
		                                intelex::grf_size<256>};

		q_->submit([&](sycl::handler& h) {
			h.parallel_for<GemmAllreduceKernelName<ElementA, ElementB>>(
			    sycl::nd_range<2>(global, local), kernel_props,
			    [=](sycl::nd_item<2>) {
			        gemm_allreduce_device(A, B, C, D, mma, alpha,
			                              ipc_signal_ptrs, ipc_data_ptrs,
			                              ipc_out_ptrs,
				                              signal_token,
				                              use_push_mode,
				                              rank, world_size, m, n, num_n_tiles,
				                              num_reducer_wgs, debug_log);
			    });
		});
	}

	bool verify(
			ElementA* block_A,
			ElementB* block_B,
			ElementC* block_D,
			Options const& options,
			int rank,
			int world_size) {
		auto& q = *q_;
		int m = options.m, n = options.n, k = options.k;
		size_t a_elems = static_cast<size_t>(m) * k;
		size_t b_elems = static_cast<size_t>(n) * k;
		size_t c_elems = static_cast<size_t>(m) * n;

		// Fill A and B with non-zero values
		q.submit([&](sycl::handler& h) {
			h.parallel_for(sycl::range<1>(a_elems), [=](sycl::id<1> i) {
				block_A[i] = static_cast<ElementA>(1.0f);
			});
		});
		q.submit([&](sycl::handler& h) {
			h.parallel_for(sycl::range<1>(b_elems), [=](sycl::id<1> i) {
				block_B[i] = static_cast<ElementB>(1.0f);
			});
		});
		q.wait();
		MPI_Barrier(MPI_COMM_WORLD);

		// Reference: single-rank GEMM → MPI_Allreduce
		ElementC* d_ref = sycl::malloc_device<ElementC>(c_elems, q);
		q.memset(d_ref, 0, c_elems * sizeof(ElementC)).wait();

		// Run a standalone GEMM (no IPC) to get reference
		auto A_ref = make_tensor(make_gmem_ptr(block_A),
		    make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
		auto B_ref = make_tensor(make_gmem_ptr(block_B),
		    make_layout(make_shape(n, k), make_stride(Int<1>{}, n)));
		auto C_ref = make_tensor(make_gmem_ptr(d_ref),
		    make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

		// Simple standalone GEMM kernel (no fusion)
		{
			namespace syclex = sycl::ext::oneapi::experimental;
			namespace intelex = sycl::ext::intel::experimental;
			syclex::properties kernel_props{syclex::sub_group_size<16>,
			                                intelex::grf_size<256>};
			auto mma = choose_tiled_mma_ar(A_ref, B_ref, C_ref);
			float a = options.alpha;
			sycl::range<2> local = {size(mma), 1};
			sycl::range<2> global = {
			    local[0] * ceil_div(n, int(get<1>(mma.tile_mnk()))),
			    local[1] * ceil_div(m, int(get<0>(mma.tile_mnk())))};
			q.submit([&](sycl::handler& h) {
				h.parallel_for(
				    sycl::nd_range<2>(global, local), kernel_props,
				    [=](sycl::nd_item<2>) {
				        // Inline simple GEMM (no allreduce)
				        auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
				        auto wg_m_ = int(item.get_group(1));
				        auto wg_n_ = int(item.get_group(0));
				        auto lid = int(item.get_local_id(0));

				        Tensor cA_ = make_identity_tensor(A_ref.shape());
				        Tensor cB_ = make_identity_tensor(B_ref.shape());
				        Tensor cC_ = make_identity_tensor(C_ref.shape());

				        auto wg_tile_ = mma.tile_mnk();
				        auto wg_coord_ = make_coord(wg_m_, wg_n_, 0);

				        Tensor gA_ = local_tile(cA_, select<0, 2>(wg_tile_), make_coord(wg_m_, _));
				        Tensor gB_ = local_tile(cB_, select<1, 2>(wg_tile_), make_coord(wg_n_, _));
				        Tensor gC_ = local_tile(cC_, wg_tile_, wg_coord_, Step<_1, _1, X>{});

				        auto copy_a_ = make_block_2d_copy_A(mma, A_ref);
				        auto copy_b_ = make_block_2d_copy_B(mma, B_ref);
				        auto copy_c_ = make_block_2d_copy_D(mma, C_ref);

				        auto thr_mma_ = mma.get_slice(lid);
				        auto thr_ca_ = copy_a_.get_slice(lid);
				        auto thr_cb_ = copy_b_.get_slice(lid);

				        auto tCrA_ = thr_mma_.partition_sg_fragment_A(gA_(_, _, 0));
				        auto tCrB_ = thr_mma_.partition_sg_fragment_B(gB_(_, _, 0));
				        auto tArA_ = thr_ca_.partition_sg_fragment_D(gA_(_, _, 0));
				        auto tBrB_ = thr_cb_.partition_sg_fragment_D(gB_(_, _, 0));
				        Tensor tAgA_ = thr_ca_.partition_S(gA_);
				        Tensor tBgB_ = thr_cb_.partition_S(gB_);
				        Tensor tCrC_ = partition_fragment_C(mma, select<0, 1>(wg_tile_));
				        Tensor tCgC_ = thr_mma_.partition_C(gC_);

				        auto pfa = make_block_2d_prefetch(copy_a_);
				        auto pfb = make_block_2d_prefetch(copy_b_);
				        auto tpA = pfa.get_slice(lid);
				        auto tpB = pfb.get_slice(lid);
				        auto pAgA_ = tpA.partition_S(gA_);
				        auto pBgB_ = tpB.partition_S(gB_);

				        int k_tiles = ceil_div(shape<1>(A_ref), get<2>(wg_tile_));
				        int kp = 0;
				        clear(tCrC_);
				        CUTE_UNROLL
				        for (; kp < 3; kp++) {
				            prefetch(pfa, pAgA_(_, _, _, kp));
				            prefetch(pfb, pBgB_(_, _, _, kp));
				        }
				        for (int kt = 0; kt < k_tiles; kt++, kp++) {
				            barrier_arrive(2);
				            copy(copy_a_, tAgA_(_, _, _, kt), tArA_);
				            copy(copy_b_, tBgB_(_, _, _, kt), tBrB_);
				            prefetch(pfa, pAgA_(_, _, _, kp));
				            prefetch(pfb, pBgB_(_, _, _, kp));
				            reorder(tArA_, tCrA_);
				            reorder(tBrB_, tCrB_);
				            gemm(mma, tCrA_, tCrB_, tCrC_);
				            barrier_wait(2);
				        }
				        auto a_elem = static_cast<ElementC>(a);
				        for (int i = 0; i < size(tCrC_); ++i) { tCrC_(i) = tCrC_(i) * a_elem; }
				        copy(copy_c_, tCrC_, tCgC_);
				    });
			});
		}
		q.wait();

		// Copy to host for MPI_Allreduce
		std::vector<ElementC> host_ref_raw(c_elems);
		q.memcpy(host_ref_raw.data(), d_ref, c_elems * sizeof(ElementC)).wait();
		sycl::free(d_ref, q);

		std::vector<float> host_ref_f32(c_elems);
		for (size_t i = 0; i < c_elems; ++i) {
			host_ref_f32[i] = static_cast<float>(host_ref_raw[i]);
		}
		std::vector<float> host_expected(c_elems);
		MPI_Allreduce(host_ref_f32.data(), host_expected.data(),
		              static_cast<int>(c_elems), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		// Run fused kernel
		ElementC* local_p2p = reinterpret_cast<ElementC*>(symm_->local_data_ptr_);
		ElementC* local_slot = local_p2p + static_cast<size_t>(rank) * c_elems;
		auto A_t = make_tensor(make_gmem_ptr(block_A),
		    make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
		auto B_t = make_tensor(make_gmem_ptr(block_B),
		    make_layout(make_shape(n, k), make_stride(Int<1>{}, n)));
		auto C_t = make_tensor(make_gmem_ptr(local_slot),
		    make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
		auto D_t = make_tensor(make_gmem_ptr(block_D),
		    make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

		q.memset(block_D, 0, c_elems * sizeof(ElementC)).wait();
		MPI_Barrier(MPI_COMM_WORLD);
		auto mma = choose_tiled_mma_ar(A_t, B_t, C_t);
		int tile_m = int(get<0>(mma.tile_mnk()));
		int tile_n = int(get<1>(mma.tile_mnk()));
		int num_n_tiles = int(ceil_div(n, tile_n));
		int num_reducer_wgs_v = 8;
		if (const char* env = std::getenv("CUTLASS_AR_NUM_REDUCER_WGS")) {
			num_reducer_wgs_v = std::max(1, std::atoi(env));
		}
		int num_reducer_rows_v = int(ceil_div(num_reducer_wgs_v, num_n_tiles));
		sycl::range<2> local = {size(mma), 1};
		sycl::range<2> global = {
		    local[0] * num_n_tiles,
		    local[1] * (int(ceil_div(m, tile_m)) + num_reducer_rows_v)};
		void** ipc_data_ptrs = reinterpret_cast<void**>(symm_->remote_data_ptrs_dev_);
		int** ipc_signal_ptrs = reinterpret_cast<int**>(symm_->remote_signal_ptrs_dev_);
		void** ipc_out_ptrs_v = ipc_out_ptrs_dev_;
		run_fused(A_t, B_t, C_t, D_t, mma, local, global, num_n_tiles,
		          ipc_data_ptrs, ipc_signal_ptrs, ipc_out_ptrs_v,
		          options.alpha, 1u, options.use_push_mode != 0,
		          rank, world_size, m, n, num_reducer_wgs_v,
		          options.debug_log != 0);
		q.wait();
		MPI_Barrier(MPI_COMM_WORLD);

		// Compare
		std::vector<ElementC> host_result(c_elems);
		q.memcpy(host_result.data(), block_D, c_elems * sizeof(ElementC)).wait();

		double max_abs_diff = 0.0, max_rel_diff = 0.0;
		size_t mismatch_count = 0;
		for (size_t i = 0; i < c_elems; ++i) {
			double ref = static_cast<double>(host_expected[i]);
			double val = static_cast<double>(static_cast<float>(host_result[i]));
			double diff = std::abs(ref - val);
			double rel = (std::abs(ref) > 1e-6) ? diff / std::abs(ref) : diff;
			max_abs_diff = std::max(max_abs_diff, diff);
			max_rel_diff = std::max(max_rel_diff, rel);
			if (rel > 1e-2) mismatch_count++;
		}

		bool passed = (mismatch_count == 0);
		printf("[rank %d] Verification %s: max_abs=%.6e, max_rel=%.6e, mismatches=%zu/%zu (expected=%.3f)\n",
		       rank, passed ? "PASSED" : "FAILED",
		       max_abs_diff, max_rel_diff, mismatch_count, c_elems,
		       host_expected[0]);
		return passed;
	}

	cutlass::Status run(
			Options const& options,
			sycl::device const& device,
			int rank,
			int world_size) {

		int m = options.m, n = options.n, k = options.k;
		if ((m % 256) != 0 || (n % 256) != 0) {
			throw std::runtime_error("gemm_allreduce requires M and N divisible by 256.");
		}
		size_t a_elems = static_cast<size_t>(m) * k;
		size_t b_elems = static_cast<size_t>(n) * k;
		size_t c_elems = static_cast<size_t>(m) * n;
		if (a_elems == 0 || b_elems == 0 || c_elems == 0) {
			throw std::runtime_error("Invalid zero-sized allocation.");
		}

		if (!q_) {
			auto ctx = sycl::context(device);
			q_ = std::make_unique<sycl::queue>(
			    ctx, device,
			    sycl::property_list{sycl::property::queue::in_order{},
			                        sycl::property::queue::enable_profiling{}});
		}

		ElementA* block_A = sycl::malloc_device<ElementA>(a_elems, *q_);
		ElementB* block_B = sycl::malloc_device<ElementB>(b_elems, *q_);
		ElementC* block_D = sycl::malloc_device<ElementC>(c_elems, *q_);

		q_->memset(block_A, 0, a_elems * sizeof(ElementA)).wait();
		q_->memset(block_B, 0, b_elems * sizeof(ElementB)).wait();
		q_->memset(block_D, 0, c_elems * sizeof(ElementC)).wait();

		auto mb = [](size_t bytes) {
			return static_cast<double>(bytes) / (1024.0 * 1024.0);
		};
		printf("[rank %d] Allocated: A=%.2f MiB, B=%.2f MiB, D=%.2f MiB\n",
		       rank,
		       mb(a_elems * sizeof(ElementA)),
		       mb(b_elems * sizeof(ElementB)),
		       mb(c_elems * sizeof(ElementC)));

		auto cleanup = [&]() {
			release_tile_signals();
			close_ipc_ptrs(*q_, opened_out_bases_);
			if (ipc_out_ptrs_dev_) {
				sycl::free(ipc_out_ptrs_dev_, *q_);
				ipc_out_ptrs_dev_ = nullptr;
			}
			if (block_A) sycl::free(block_A, *q_);
			if (block_B) sycl::free(block_B, *q_);
			if (block_D) sycl::free(block_D, *q_);
		};

		// Build A/B and a metadata-only C tensor for MMA tile selection
		auto A = make_tensor(make_gmem_ptr(block_A),
		    make_layout(make_shape(m, k), make_stride(k, Int<1>{})));
		auto B = make_tensor(make_gmem_ptr(block_B),
		    make_layout(make_shape(n, k), make_stride(Int<1>{}, n)));
		auto C_meta = make_tensor(make_gmem_ptr(block_D),
		    make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

		// Initialize tile-level IPC signals and create SymmMemory (once)
		initialize_tile_signals(m, n, k, rank, world_size, A, B, C_meta);

		// Exchange IPC for output buffer D (needed for RS+AG allgather phase)
		{
			auto peer_out = exchange_ipc_ptrs(block_D, rank, world_size, *q_, opened_out_bases_);
			ipc_out_ptrs_dev_ = sycl::malloc_device<void*>(world_size, *q_);
			q_->memcpy(ipc_out_ptrs_dev_, peer_out.data(), world_size * sizeof(void*)).wait();

			// Make peer output buffers resident on local device
			auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_->get_context());
			auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_->get_device());
			for (int r = 0; r < world_size; ++r) {
				if (r != rank) {
					zeContextMakeMemoryResident(ze_ctx, ze_dev, peer_out[r],
					    c_elems * sizeof(ElementC));
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// Build C/D tensors (reused for all iterations)
		ElementC* local_p2p = reinterpret_cast<ElementC*>(symm_->local_data_ptr_);
		ElementC* local_slot = local_p2p + static_cast<size_t>(rank) * c_elems;
		auto C = make_tensor(make_gmem_ptr(local_slot),
		    make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
		auto D = make_tensor(make_gmem_ptr(block_D),
		    make_layout(make_shape(m, n), make_stride(n, Int<1>{})));

		// Precompute fused launch config once and reuse in loops
		auto mma = choose_tiled_mma_ar(A, B, C);
		int tile_m = int(get<0>(mma.tile_mnk()));
		int tile_n = int(get<1>(mma.tile_mnk()));
		int num_m_tiles = int(ceil_div(m, tile_m));
		int num_n_tiles = int(ceil_div(n, tile_n));

		// Number of reducer WGs: configurable via CUTLASS_AR_NUM_REDUCER_WGS env var
		// Must be limited to avoid deadlock: GEMM WGs + reducer WGs must fit in
		// hardware simultaneously, otherwise reducers block GEMM from running.
		int num_reducer_wgs = 8;  // default: safe for BMG (8 GEMM + 8 reducer = 16 WGs)
		if (const char* env = std::getenv("CUTLASS_AR_NUM_REDUCER_WGS")) {
			num_reducer_wgs = std::max(1, std::atoi(env));
		}
		// Reducer WGs are appended as extra rows in dim 1.
		// Each row has num_n_tiles WGs in dim 0, so we need enough rows.
		int num_reducer_rows = int(ceil_div(num_reducer_wgs, num_n_tiles));
		// Actual number of reducer WGs launched (may be slightly more than requested)
		int actual_reducer_wgs_launched = num_reducer_rows * num_n_tiles;

		sycl::range<2> local = {size(mma), 1};
		sycl::range<2> global = {
		    local[0] * num_n_tiles,
		    local[1] * (num_m_tiles + num_reducer_rows)};
		void** ipc_data_ptrs = reinterpret_cast<void**>(symm_->remote_data_ptrs_dev_);
		int** ipc_signal_ptrs = reinterpret_cast<int**>(symm_->remote_signal_ptrs_dev_);
		void** ipc_out_ptrs = ipc_out_ptrs_dev_;

		if (rank == 0) {
			printf("[rank %d] Grid: %d GEMM WGs + %d reducer WGs (%d requested), "
			       "subgroups_per_reducer_wg=%d\n",
			       rank, num_m_tiles * num_n_tiles, actual_reducer_wgs_launched,
			       num_reducer_wgs, int(size(mma)) / 16);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "[rank " << rank << "] initialization complete" << std::endl;

		// Verification
		if (options.verify != 0 && !verify(block_A, block_B, block_D, options, rank, world_size)) {
			std::cerr << "[rank " << rank << "] verification failed!" << std::endl;
			cleanup();
			return cutlass::Status::kErrorInternal;
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// Warmup
		constexpr int kWarmupIters = 10;
		uint32_t signal_token = (options.verify != 0) ? 2u : 1u;
		auto next_signal_token = [&]() {
			uint32_t token = signal_token++;
			if (signal_token == 0) signal_token = 1u;
			return token;
		};
		bool enable_debug_log = (options.debug_log != 0);
		std::cout << "[rank " << rank << "] warmup start (" << kWarmupIters << " iters)" << std::endl;
		for (int iter = 0; iter < kWarmupIters; ++iter) {
			// Only log on the first warmup iteration to avoid flooding
			run_fused(A, B, C, D, mma, local, global, num_n_tiles,
			          ipc_data_ptrs, ipc_signal_ptrs, ipc_out_ptrs,
			          options.alpha, next_signal_token(), options.use_push_mode != 0,
			          rank, world_size, m, n, num_reducer_wgs,
			          enable_debug_log && (iter == 0));
		}
		q_->wait();
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "[rank " << rank << "] warmup done" << std::endl;

		// Benchmark
		auto benchmark_start = std::chrono::high_resolution_clock::now();
		for (int iter = 0; iter < options.iterations; ++iter) {
			run_fused(A, B, C, D, mma, local, global, num_n_tiles,
			          ipc_data_ptrs, ipc_signal_ptrs, ipc_out_ptrs,
			          options.alpha, next_signal_token(), options.use_push_mode != 0,
			          rank, world_size, m, n, num_reducer_wgs);
		}
		auto ev_after = q_->ext_oneapi_submit_barrier();
		q_->wait();
		auto benchmark_stop = std::chrono::high_resolution_clock::now();

		MPI_Barrier(MPI_COMM_WORLD);

		double total_ms = std::chrono::duration<double, std::milli>(benchmark_stop - benchmark_start).count();
		double avg_ms = total_ms / options.iterations;
		double tflops = (2.0 * m * n * k) * 1e-12;
		std::cout << "[" << rank << "] Problem Size: " << m << 'x' << n << 'x' << k
		          << ", TP=" << world_size << std::endl;
		printf("[%d] GEMM + Allreduce (host): [%4.3f]TFlop/s  (%6.4f)ms\n",
		       rank, tflops / (avg_ms / 1000.0), avg_ms);

		cleanup();
		return cutlass::Status::kSuccess;
	}
};

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int world_size = 1;
	int rank = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	Options options;
	options.parse(argc, argv);

	if (options.help) {
		options.print_usage(std::cout) << std::endl;
		MPI_Finalize();
		return 0;
	}
	if (options.error) {
		MPI_Finalize();
		return -1;
	}

	auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	if (devices.empty()) {
		if (rank == 0) std::cerr << "No GPU devices found" << std::endl;
		MPI_Finalize();
		return 1;
	}
	if (static_cast<size_t>(rank) >= devices.size()) {
		std::cerr << "Rank " << rank << " requires GPU device[" << rank << "], but only "
		          << devices.size() << " devices are available" << std::endl;
		MPI_Finalize();
		return 1;
	}

	auto device = devices[rank];

	ExampleRunner runner;
	try {
		CUTLASS_CHECK(runner.run(options, device, rank, world_size));
	} catch (std::exception const& e) {
		std::cerr << "[rank " << rank << "] " << e.what() << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Finalize();
	return 0;
}
