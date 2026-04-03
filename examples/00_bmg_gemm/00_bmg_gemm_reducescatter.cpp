
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include <mpi.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cute/tensor.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

#include "symm.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "sycl_common.hpp"

using namespace cute;

struct Options {
	bool help = false;
	bool error = false;
	int m = 5120;
	int n = 4096;
	int k = 4096;
	int l = 1;
	int iterations = 20;
	int debug_log = 1;
	float alpha = 1.0f;
	float beta = 0.0f;

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
		cmd.get_cmd_line_argument("m", m, 5120);
		cmd.get_cmd_line_argument("n", n, 4096);
		cmd.get_cmd_line_argument("k", k, 4096);
		cmd.get_cmd_line_argument("l", l, 1);
		cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
		cmd.get_cmd_line_argument("beta", beta, 0.0f);
		cmd.get_cmd_line_argument("iterations", iterations, 20);
		cmd.get_cmd_line_argument("debug_log", debug_log, 1);
	}

	std::ostream& print_usage(std::ostream& out) const {
		out << "BMG GEMM Reduce-Scatter Example\n\n"
			<< "  --m=<int>         M extent (global, must be divisible by TP)\n"
			<< "  --n=<int>         N extent\n"
			<< "  --k=<int>         K extent\n"
			<< "  --l=<int>         batch count\n"
			<< "  --iterations=<int>\n"
			<< "  --debug_log=<int> 0/1 progress logs (default 1)\n\n";
		return out;
	}
};

template <class Gemm>
struct ExampleRunner {
	using StrideA = typename Gemm::GemmKernel::StrideA;
	using StrideB = typename Gemm::GemmKernel::StrideB;
	using StrideC = typename Gemm::GemmKernel::StrideC;
	using StrideD = typename Gemm::GemmKernel::StrideD;

	using ElementA = typename Gemm::ElementA;
	using ElementB = typename Gemm::ElementB;
	using ElementAccumulator = typename Gemm::ElementAccumulator;

	using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
	using ElementC = typename Gemm::ElementC;
	using ElementOutput = typename CollectiveEpilogue::ElementOutput;
	using ElementCompute = typename CollectiveEpilogue::ElementCompute;

	using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

	StrideA stride_A;
	StrideB stride_B;
	StrideC stride_C;
	StrideD stride_D;
	std::unique_ptr<sycl::queue> current_q_;
	std::unique_ptr<sycl::queue> tmp_q_;
	Gemm gemm_op_;
	// IPC pipeline state — one slot per remote step (step in 1..world_size-1)
	ElementOutput* local_p2p_ = nullptr;
	ElementOutput** remote_p2p_ptrs_ = nullptr;
	ElementOutput* stacked_partials_ = nullptr;
	std::vector<void*> opened_p2p_bases_;
	size_t chunk_elements_ = 0;
	size_t total_elements_ = 0;

	void initialize(
			ElementA* block_A,
			ElementB* block_B,
			ElementC* block_C,
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			sycl::device const& device,
			int rank,
			int world_size) {
		auto log_init = [&](char const* msg) {
			if (options.debug_log) {
				std::cout << "[rank " << rank << "] [init] " << msg << std::endl;
			}
		};

		log_init("initialize begin");
		int local_rows = options.m / world_size;

		if (!current_q_) {
			log_init("creating current_q");
			auto ctx = sycl::context(device);
			current_q_ = std::make_unique<sycl::queue>(
					ctx,
					device,
					sycl::property_list{sycl::property::queue::in_order{}});
			log_init("current_q created");
		}

		auto ctx = current_q_->get_context();
		if (!tmp_q_) {
			log_init("creating tmp_q");
			tmp_q_ = std::make_unique<sycl::queue>(
					ctx,
					device,
					sycl::property_list{sycl::property::queue::in_order{}});
			log_init("tmp_q created");
		}

		stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
		stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
		stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
		stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

		chunk_elements_ = static_cast<size_t>(local_rows) * options.n * options.l;
		total_elements_ = static_cast<size_t>(options.m) * options.n * options.l;

		if (!local_p2p_) {
			log_init("allocating IPC buffers");
			size_t slots = static_cast<size_t>(world_size - 1);
			local_p2p_ = sycl::malloc_device<ElementOutput>(slots * chunk_elements_, *current_q_);
			stacked_partials_ = sycl::malloc_device<ElementOutput>(total_elements_, *current_q_);
			current_q_->memset(local_p2p_, 0, slots * chunk_elements_ * sizeof(ElementOutput)).wait();
			current_q_->memset(stacked_partials_, 0, total_elements_ * sizeof(ElementOutput)).wait();
			remote_p2p_ptrs_ = exchange_ipc_ptrs(local_p2p_, rank, world_size, *current_q_, opened_p2p_bases_);
			log_init("IPC buffers allocated");
		}

		log_init("before block_A memset");
		current_q_->memset(block_A, 0, static_cast<size_t>(options.m) * options.k * options.l * sizeof(ElementA)).wait();
		log_init("before block_B memset");
		current_q_->memset(block_B, 0, static_cast<size_t>(options.n) * options.k * options.l * sizeof(ElementB)).wait();
		log_init("before block_C memset");
		current_q_->memset(block_C, 0, static_cast<size_t>(options.m) * options.n * options.l * sizeof(ElementC)).wait();
		log_init("initialize end");
	}

	cutlass::Status produce_chunk(
			sycl::queue& queue,
			Gemm& gemm_op,
			ElementA* block_A,
			ElementB* block_B,
			ElementC* block_C,
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			int producer_rank,
			int local_rows,
			ElementOutput* dst_ptr) {

		ProblemShapeType subproblem{local_rows, options.n, options.k, options.l};

		auto sub_stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(local_rows, options.k, options.l));
		auto sub_stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(local_rows, options.n, options.l));
		auto sub_stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(local_rows, options.n, options.l));

		size_t a_off = static_cast<size_t>(producer_rank) * local_rows * options.k * options.l;
		size_t c_off = static_cast<size_t>(producer_rank) * local_rows * options.n * options.l;

		typename Gemm::GemmKernel::Arguments args{
			cutlass::gemm::GemmUniversalMode::kGemm,
			subproblem,
			{block_A + a_off, sub_stride_A, block_B, stride_B},
			{{options.alpha, options.beta}, block_C + c_off, sub_stride_C, dst_ptr, sub_stride_D},
			hw_info};

		if (Gemm::get_workspace_size(args) != 0) return cutlass::Status::kErrorInternal;

		auto st = gemm_op.can_implement(args);
		if (st != cutlass::Status::kSuccess) return st;
		st = gemm_op.initialize(args, nullptr, &queue);
		if (st != cutlass::Status::kSuccess) return st;
		return gemm_op.run(&queue);
	}

	static void enqueue_stream_bias(sycl::queue& q) {
		q.submit([&](sycl::handler& h) {
			h.single_task([=]() {
				volatile int delay = 0;
				for (int i = 0; i < 4096; ++i) {
					delay += i;
				}
				(void)delay;
			});
		});
	}

	struct reduce_scatter {
		ExampleRunner& runner;

		void operator()(
				sycl::queue& q,
				ElementA* block_A,
				ElementB* block_B,
				ElementC* block_C,
				Options const& options,
				cutlass::KernelHardwareInfo const& hw_info,
				int rank,
				int world_size) const {
			int local_rows = options.m / world_size;
			size_t chunk_e = runner.chunk_elements_;
			size_t local_off = static_cast<size_t>(rank) * chunk_e;
			size_t total_bytes = runner.total_elements_ * sizeof(ElementOutput);
			size_t chunk_bytes = chunk_e * sizeof(ElementOutput);

			if (runner.local_p2p_ == nullptr || runner.remote_p2p_ptrs_ == nullptr) {
				throw std::runtime_error("IPC pointers are null.");
			}

			auto log = [&](char const* msg) {
				if (options.debug_log) {
					std::cout << "[rank " << rank << "] " << msg << std::endl;
				}
			};

			auto& current_q = *runner.current_q_;
			auto& tmp_q = *runner.tmp_q_;

			current_q.memset(runner.stacked_partials_, 0, total_bytes).wait();

			for (int step = 1; step < world_size; ++step) {
				int remote_rank = (rank + step) % world_size;
				int channel = 0; //step % 2;
				auto& queue = (channel == 0) ? current_q : tmp_q;
				// if (options.debug_log) {
				// 	std::cout << "[rank " << rank << "] step=" << step
				// 	          << " remote_rank=" << remote_rank
				// 	          << " channel=" << channel << std::endl;
				// }

				// Compute shard for remote_rank → write to local p2p slot
				ElementOutput* p2p_dst = runner.local_p2p_ + static_cast<size_t>(step - 1) * chunk_e;
				auto st = runner.produce_chunk(queue, runner.gemm_op_, block_A, block_B, block_C,
				                              options, hw_info, remote_rank, local_rows, p2p_dst);
				if (st != cutlass::Status::kSuccess)
					throw std::runtime_error("produce_chunk (remote shard) failed.");

				// Pull from remote_rank: the shard they computed for us
				int pull_step = ((rank - remote_rank) + world_size) % world_size;
				ElementOutput* remote_src = runner.remote_p2p_ptrs_[remote_rank]
				                          + static_cast<size_t>(pull_step - 1) * chunk_e;
				queue.memcpy(
					runner.stacked_partials_ + static_cast<size_t>(remote_rank) * chunk_e,
					remote_src,
					chunk_bytes);
			}

			// Local shard written directly into stacked_partials[rank].
			auto st = runner.produce_chunk(current_q, runner.gemm_op_, block_A, block_B, block_C,
			                              options, hw_info, rank, local_rows,
			                              runner.stacked_partials_ + local_off);
			if (st != cutlass::Status::kSuccess)
				throw std::runtime_error("produce_chunk (local shard) failed.");

			current_q.wait();
			tmp_q.wait();

			// ---- Phase 3: Offline reduce on GPU -----------------------------------------
			// Reduce stacked_partials[0..world_size-1] into stacked_partials[0]
			{
				ElementOutput* sp = runner.stacked_partials_;
				int ws = world_size;
				size_t ce = chunk_e;
				current_q.submit([&](sycl::handler& h) {
					h.parallel_for(sycl::range<1>(chunk_e), [=](sycl::id<1> idx) {
						ElementOutput acc = ElementOutput(0);
						for (int r = 0; r < ws; ++r)
							acc += sp[static_cast<size_t>(r) * ce + idx[0]];
						sp[idx[0]] = acc;
					});
				});
				current_q.wait();
			}
			log("reduce phase done");
		}
	};

	void run_iteration(
			sycl::queue& q,
			ElementA* block_A,
			ElementB* block_B,
			ElementC* block_C,
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			int rank,
			int world_size) {

		reduce_scatter op{*this};
		if (options.debug_log) {
			std::cout << "[rank " << rank << "] entering reduce_scatter" << std::endl;
		}
		op(q, block_A, block_B, block_C, options, hw_info, rank, world_size);
	}

	cutlass::Status run(
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			sycl::device const& device,
			int rank,
			int world_size) {

		if (options.m % world_size != 0)
			throw std::runtime_error("GEMM+reduce-scatter requires M divisible by world_size.");

		size_t a_elems = static_cast<size_t>(options.m) * options.k * options.l;
		size_t b_elems = static_cast<size_t>(options.n) * options.k * options.l;
		size_t c_elems = static_cast<size_t>(options.m) * options.n * options.l;

		if (!current_q_) {
			auto queue_context = sycl::context(device);
			current_q_ = std::make_unique<sycl::queue>(
					queue_context,
					device,
					sycl::property_list{sycl::property::queue::in_order{}});
		}

		ElementA* block_A = sycl::malloc_device<ElementA>(a_elems, *current_q_);
		ElementB* block_B = sycl::malloc_device<ElementB>(b_elems, *current_q_);
		ElementC* block_C = sycl::malloc_device<ElementC>(c_elems, *current_q_);
		if (block_A == nullptr || block_B == nullptr || block_C == nullptr) {
			throw std::runtime_error("Device allocation failed.");
		}

		auto cleanup = [&]() {
			if (block_A) sycl::free(block_A, *current_q_);
			if (block_B) sycl::free(block_B, *current_q_);
			if (block_C) sycl::free(block_C, *current_q_);
			close_ipc_ptrs(*current_q_, opened_p2p_bases_);
			if (remote_p2p_ptrs_) { sycl::free(remote_p2p_ptrs_, *current_q_); remote_p2p_ptrs_ = nullptr; }
			if (local_p2p_)       { sycl::free(local_p2p_,       *current_q_); local_p2p_ = nullptr; }
			if (stacked_partials_){ sycl::free(stacked_partials_, *current_q_); stacked_partials_ = nullptr; }
		};

		initialize(block_A, block_B, block_C, options, hw_info, device, rank, world_size);
		if (options.debug_log) {
			std::cout << "[rank " << rank << "] initialization complete" << std::endl;
		}

		// warmup
		constexpr int kWarmupIters = 10;
		if (options.debug_log && rank == 0) {
			std::cout << "[rank " << rank << "] warmup start (" << kWarmupIters << " iters)" << std::endl;
		}
		for (int iter = 0; iter < kWarmupIters; ++iter) {
			run_iteration(*current_q_, block_A, block_B, block_C, options, hw_info, rank, world_size);
		}
		if (options.debug_log && rank == 0) {
			std::cout << "[rank " << rank << "] warmup done" << std::endl;
		}

		current_q_->wait();
		MPI_Barrier(MPI_COMM_WORLD); // ensure all ranks have finished warmup before starting benchmark iterations

		// benchmark
		double total_ms = 0.0;
		for (int iter = 0; iter < options.iterations; ++iter) {
			if (options.debug_log) {
				std::cout << "[rank " << rank << "] benchmark iteration " << iter << " start" << std::endl;
			}
			auto start = std::chrono::high_resolution_clock::now();
			run_iteration(*current_q_, block_A, block_B, block_C, options, hw_info, rank, world_size);
			auto stop = std::chrono::high_resolution_clock::now();
			double local_elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();

			current_q_->wait();
			double max_rank_ms = 0.0;
			MPI_Allreduce(&local_elapsed_ms, &max_rank_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			total_ms += max_rank_ms;
			if (options.debug_log) {
				std::cout << "[rank " << rank << "] benchmark iteration " << iter << " end" << std::endl;
			}
		}

		if (rank == 0) {
			double avg_ms = total_ms / options.iterations;
			double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
			std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k
			          << 'x' << options.l << ", TP=" << world_size << std::endl;
			printf("Pipelined GEMM + Reduce-Scatter: [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / (avg_ms / 1000.0), avg_ms);
		}

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
		if (rank == 0) {
			std::cerr << "No GPU devices found" << std::endl;
		}
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

	cutlass::KernelHardwareInfo hw_info;
	hw_info.device_id = rank;
	hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

	using ElementAccumulator = float;
	using ElementComputeEpilogue = float;
	using ElementInputA = bfloat16_t;
	using ElementInputB = bfloat16_t;
	using ElementOutput = float;

	using LayoutA = cutlass::layout::RowMajor;
	using LayoutB = cutlass::layout::RowMajor;
	using LayoutC = cutlass::layout::RowMajor;
	using LayoutD = cutlass::layout::RowMajor;

	using GmemTiledCopyA = void;
	using GmemTiledCopyB = void;
	using TileShape = Shape<_256, _256, _32>;
	using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<TileShape>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
	constexpr int PipelineStages = 2;
	using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
	using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;
	using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
			ElementOutput,
			ElementComputeEpilogue,
			ElementAccumulator,
			ElementAccumulator,
			cutlass::FloatRoundStyle::round_to_nearest>;
	using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
			EpilogueDispatchPolicy,
			EpilogueOp,
			TileShape,
			decltype(tile_shape(TiledMma()))>;
	using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
			EpilogueDispatchPolicy,
			TileShape,
			void,
			ElementAccumulator,
			cutlass::gemm::TagToStrideC_t<LayoutC>,
			ElementOutput,
			cutlass::gemm::TagToStrideC_t<LayoutD>,
			FusionCallbacks,
			void,
			void>;
	using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
			GEMMDispatchPolicy,
			TileShape,
			ElementInputA,
			cutlass::gemm::TagToStrideA_t<LayoutA>,
			ElementInputB,
			cutlass::gemm::TagToStrideB_t<LayoutB>,
			TiledMma,
			GmemTiledCopyA,
			void,
			void,
			cute::identity,
			GmemTiledCopyB,
			void,
			void,
			cute::identity>;
	using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
			Shape<int, int, int, int>,
			CollectiveMainloop,
			CollectiveEpilogue>;
	using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

	ExampleRunner<Gemm> runner;
	try {
		CUTLASS_CHECK(runner.run(options, hw_info, device, rank, world_size));
	} catch (std::exception const& e) {
		std::cerr << "[rank " << rank << "] " << e.what() << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Finalize();
	return 0;
}
