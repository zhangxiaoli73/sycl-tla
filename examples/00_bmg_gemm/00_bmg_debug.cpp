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

#include "cutlass/util/command_line.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "symm.hpp"
#include "sycl_common.hpp"

using namespace cute;

struct Options {
	bool help = false;
	bool error = false;
	int m = 8192;
	int n = 1536;
	int k = 4096;
	int l = 1;
	int iterations = 20;
	int debug_log = 0;
	int gemm_disable = 0;
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
		cmd.get_cmd_line_argument("gemm_disable", gemm_disable, 0);
	}

	std::ostream& print_usage(std::ostream& out) const {
		out << "BMG allgather + GEMM Example\n\n"
				<< "  --m=<int>        global M, must be divisible by TP\n"
				<< "  --n=<int>        N\n"
				<< "  --k=<int>        K\n"
				<< "  --l=<int>        batch count\n"
				<< "  --iterations=<int>\n"
				<< "  --debug_log=<int> 0/1 progress logs (default 1)\n"
				<< "  --gemm_disable=<int> 0/1 skip GEMM in allgather (default 0)\n\n";
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
	ProblemShapeType shard_problem_{};
	StrideA shard_stride_A{};
	StrideC shard_stride_C{};
	StrideD shard_stride_D{};
	std::unique_ptr<sycl::queue> current_q_;
	std::unique_ptr<sycl::queue> tmp_q_;
	std::unique_ptr<SymmMemory> symm_;
	Gemm gemm_op_;
	bool gemm_initialized_ = false;
	uint64_t seed = 0;

	void initialize(ElementA* local_A,
					ElementB* B,
					ElementOutput* final_C,
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
		int local_m = options.m / world_size;
		int n = options.n;
		int k = options.k;

		if (!current_q_) {
			log_init("creating current_q");
			auto ctx = sycl::context(device);
			current_q_ = std::make_unique<sycl::queue>(
					ctx,
					device,
					sycl::property_list{sycl::property::queue::in_order{}});
			log_init("current_q created");
		}

		if (!symm_) {
			log_init("creating SymmMemory");
			symm_ = std::make_unique<SymmMemory>(local_m, n, k, rank, world_size, *current_q_, 8);
			if (options.debug_log) {
				std::cout << "[rank " << rank << "] SymmMemory initialized" << std::endl;
			}
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

		stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(local_m, k, 1));
		stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
		stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(local_m, n, 1));
		stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(local_m, n, 1));
		shard_problem_ = ProblemShapeType{local_m, n, k, 1};
		shard_stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(local_m, k, 1));
		shard_stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(local_m, n, 1));
		shard_stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(local_m, n, 1));

		log_init("before local_A memset");
		current_q_->memset(local_A, 0, static_cast<size_t>(local_m) * k * sizeof(ElementA)).wait();
		log_init("after local_A memset");
		log_init("before B memset");
		current_q_->memset(B, 0, static_cast<size_t>(n) * k * sizeof(ElementB)).wait();
		log_init("after B memset");
		log_init("input buffers initialized");
		if (final_C == nullptr) {
			throw std::runtime_error("final_C is null before q.fill; device allocation likely failed.");
		}
		log_init("before final_C fill");
		current_q_->memset(final_C, 0, static_cast<size_t>(options.m) * n * sizeof(ElementOutput)).wait();
		log_init("after final_C fill");

		ElementA* gathered_A = reinterpret_cast<ElementA*>(symm_->local_data_ptr());
		if (gathered_A == nullptr) {
			throw std::runtime_error("symm local_data_ptr is null.");
		}

		typename Gemm::GemmKernel::Arguments template_args{
				cutlass::gemm::GemmUniversalMode::kGemm,
				shard_problem_,
				{gathered_A, shard_stride_A, B, stride_B},
				{{options.alpha, options.beta}, final_C, shard_stride_C, final_C, shard_stride_D},
				hw_info};

		if (!gemm_initialized_) {
			log_init("before can_implement");
			auto st = gemm_op_.can_implement(template_args);
			if (st != cutlass::Status::kSuccess) {
				throw std::runtime_error("GEMM cannot implement shard args.");
			}
			log_init("before gemm initialize");
			st = gemm_op_.initialize(template_args, nullptr, current_q_.get());
			if (st != cutlass::Status::kSuccess) {
				throw std::runtime_error("GEMM initialize failed.");
			}
			gemm_initialized_ = true;
			log_init("after gemm initialize");
			if (options.debug_log) {
				std::cout << "[rank " << rank << "] GEMM operator initialized" << std::endl;
			}
		}
		log_init("initialize end");
	}

	cutlass::Status run_shard_gemm(
			sycl::queue& queue,
			ElementA* a_ptr,
			ElementB* b_ptr,
			ElementC* c_ptr,
			ElementOutput* d_ptr,
			ElementCompute alpha,
			ElementCompute beta,
			cutlass::KernelHardwareInfo const& hw_info) {

		typename Gemm::GemmKernel::Arguments args{
				cutlass::gemm::GemmUniversalMode::kGemm,
				shard_problem_,
				{a_ptr, shard_stride_A, b_ptr, stride_B},
				{{alpha, beta}, c_ptr, shard_stride_C, d_ptr, shard_stride_D},
				hw_info};

		if (Gemm::get_workspace_size(args) != 0) {
			return cutlass::Status::kErrorInternal;
		}

		if (!gemm_initialized_) {
			return cutlass::Status::kErrorInternal;
		}

		auto st = gemm_op_.update(args, nullptr);
		if (st != cutlass::Status::kSuccess) {
			return st;
		}
		return gemm_op_.run(&queue);
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

	struct allgather_gemm {
		ExampleRunner& runner;

		std::pair<double, double> operator()(
				sycl::queue& queue,
				ElementA* local_A,
				ElementB* B,
				ElementOutput* final_C,
				SymmMemory& symm,
				Options const& options,
				cutlass::KernelHardwareInfo const& hw_info,
				sycl::context const&,
				sycl::device const&,
				int rank,
				int world_size) const {
			int local_m = options.m / world_size;
			size_t shard_a_elems = static_cast<size_t>(local_m) * options.k;
			size_t shard_c_elems = static_cast<size_t>(local_m) * options.n;
			size_t shard_a_bytes = shard_a_elems * sizeof(ElementA);
			ElementA* gathered_A = reinterpret_cast<ElementA*>(symm.local_data_ptr());
			auto remote_data_ptrs = reinterpret_cast<ElementA**>(symm.remote_data_ptrs());
			if (gathered_A == nullptr || remote_data_ptrs == nullptr) {
				throw std::runtime_error("SymmMemory IPC pointers are null.");
			}

			auto& current_q = *runner.current_q_;

			double total_copy_us = 0.0;
			double total_gemm_us = 0.0;

			for (int step = 0; step < world_size; ++step) {
				int remote_rank = (rank + step) % world_size;

				ElementA* remote_src = remote_data_ptrs[remote_rank];
				ElementA* local_dst = gathered_A + static_cast<size_t>(remote_rank) * shard_a_elems;

				auto t0 = std::chrono::high_resolution_clock::now();
				queue.memcpy(local_dst, remote_src, shard_a_bytes);
				auto t1 = std::chrono::high_resolution_clock::now();

				auto t2 = t1;
				if (!options.gemm_disable) {
					auto st = runner.run_shard_gemm(
							queue,
							local_A,
							B,
							final_C + static_cast<size_t>(remote_rank) * shard_c_elems,
							final_C + static_cast<size_t>(remote_rank) * shard_c_elems,
							options.alpha,
							options.beta,
							hw_info);
					t2 = std::chrono::high_resolution_clock::now();
				}

				total_copy_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
				total_gemm_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
			}

			return std::make_pair(total_copy_us, total_gemm_us);
		}
	};

	std::pair<double, double> run_iteration(
			sycl::queue& q,
			ElementA* local_A,
			ElementB* B,
			ElementOutput* final_C,
			SymmMemory& symm,
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			sycl::context const& ctx,
			sycl::device const& dev,
			int rank,
			int world_size) {

		allgather_gemm op{*this};
		if (options.debug_log) {
			std::cout << "[rank " << rank << "] entering allgather_gemm" << std::endl;
		}
		return op(q, local_A, B, final_C, symm, options, hw_info, ctx, dev, rank, world_size);
	}

	cutlass::Status run(
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			sycl::device const& device,
			int rank,
			int world_size) {

		if (options.m % world_size != 0) {
			throw std::runtime_error("allgather+gemm requires M divisible by world_size.");
		}

		int local_m = options.m / world_size;
		size_t local_a_elems = static_cast<size_t>(local_m) * options.k;
		size_t b_elems = static_cast<size_t>(options.n) * options.k;
		size_t full_c_elems = static_cast<size_t>(options.m) * options.n;
		if (local_a_elems == 0 || b_elems == 0 || full_c_elems == 0) {
			throw std::runtime_error("Invalid zero-sized allocation request. Check m/n/k/world_size values.");
		}
		if (!device.get_info<sycl::info::device::usm_device_allocations>()) {
			throw std::runtime_error("Selected SYCL device does not support USM device allocations.");
		}

		if (!current_q_) {
			auto queue_context = sycl::context(device);
			current_q_ = std::make_unique<sycl::queue>(
					queue_context,
					device,
					sycl::property_list{sycl::property::queue::in_order{}});
		}

		sycl::context ctx = current_q_->get_context();
		ElementA* local_A = sycl::malloc_device<ElementA>(local_a_elems, *current_q_);
		ElementB* B = sycl::malloc_device<ElementB>(b_elems, *current_q_);
		ElementOutput* final_C = sycl::malloc_device<ElementOutput>(full_c_elems, *current_q_);
		if (local_A == nullptr || B == nullptr || final_C == nullptr) {
			auto mb = [](size_t bytes) {
				return static_cast<double>(bytes) / (1024.0 * 1024.0);
			};
			throw std::runtime_error(
				"Device allocation failed: local_A=" + std::to_string(mb(local_a_elems * sizeof(ElementA))) +
				" MiB, B=" + std::to_string(mb(b_elems * sizeof(ElementB))) +
				" MiB, final_C=" + std::to_string(mb(full_c_elems * sizeof(ElementOutput))) + " MiB.");
		}

		auto cleanup = [&]() {
			if (local_A) sycl::free(local_A, *current_q_);
			if (B) sycl::free(B, *current_q_);
			if (final_C) sycl::free(final_C, *current_q_);
		};

		initialize(local_A, B, final_C,
				options, hw_info, device, rank, world_size);
		if (options.debug_log) {
			std::cout << "[rank " << rank << "] initialization complete" << std::endl;
		}

		// warmup
		constexpr int kWarmupIters = 10;
		if (options.debug_log && rank == 0) {
			std::cout << "[rank " << rank << "] warmup start (" << kWarmupIters << " iters)" << std::endl;
		}
		for (int iter = 0; iter < kWarmupIters; ++iter) {
			run_iteration(*current_q_, local_A, B, final_C, *symm_, options, hw_info, ctx, device, rank, world_size);
		}
		if (options.debug_log && rank == 0) {
			std::cout << "[rank " << rank << "] warmup done" << std::endl;
		}

		current_q_->wait();
		MPI_Barrier(MPI_COMM_WORLD); // ensure all ranks have finished warmup before starting benchmark iterations

		// benchmark
		double total_copy_us = 0.0;
		double total_gemm_us = 0.0;
		for (int iter = 0; iter < options.iterations; ++iter) {
            MPI_Barrier(MPI_COMM_WORLD);
			if (options.debug_log) {
				std::cout << "[rank " << rank << "] benchmark iteration " << iter << " start" << std::endl;
			}
			auto [copy_us, gemm_us] = run_iteration(*current_q_, local_A, B, final_C, *symm_, options, hw_info, ctx, device, rank, world_size);

			total_copy_us += copy_us;
			total_gemm_us += gemm_us;

			current_q_->wait();

			std::cout << "[rank " << rank << "] iter=" << iter
			          << "  copy_api: " << copy_us << " us"
			          << "  gemm_api: " << gemm_us << " us" << std::endl;
		}

		if (true) {
			double avg_copy_us = total_copy_us / options.iterations;
			double avg_gemm_us = total_gemm_us / options.iterations;
			std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k
			          << ", TP=" << world_size << std::endl;
			printf("  Avg memcpy API: %.1f us,  Avg run_shard_gemm API: %.1f us\n", avg_copy_us, avg_gemm_us);
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
    std::cout << "Available SYCL GPU devices:" << std::endl;
	for (size_t i = 0; i < devices.size(); ++i) {
		std::cout << "  Device[" << i << "]: " << devices[i].get_info<sycl::info::device::name>() << std::endl;
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
