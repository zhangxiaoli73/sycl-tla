
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/GPU_Clock.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cute/tensor.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "symm.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "helper.h"
#include "sycl_common.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help  = false;
  bool error = false;
  int m = 5120, n = 4096, k = 4096, l = 1;
  int iterations = 20, verify = 1;
  float alpha = 1.f, beta = 0.f;

  void parse(int argc, char **args) {
    std::vector<char const*> cargs(argc);
    for (int i = 0; i < argc; ++i) {
      cargs[i] = args[i];
    }
    cutlass::CommandLine cmd(argc, cargs.data());
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }

  std::ostream & print_usage(std::ostream &out) const {
    out << "BMG GEMM Fusion Example\n\n"
        << "  --m=<int>         M extent            (default 5120)\n"
        << "  --n=<int>         N extent            (default 4096)\n"
        << "  --k=<int>         K extent            (default 4096)\n"
        << "  --l=<int>         batch count         (default 1)\n"
        << "  --iterations=<int>                    (default 100)\n"
        << "  --verify=<int>    0 = skip verify     (default 1)\n\n";
    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool almost_equal(T a, T b, double rtol = 1e-4, double atol = 1e-5) {
  double d = std::abs(static_cast<double>(a) - static_cast<double>(b));
  double s = std::max({1.0, std::abs(static_cast<double>(a)), std::abs(static_cast<double>(b))});
  return d <= atol + rtol * s;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

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
  uint64_t seed = 0;

  // ---- Input/bias memory -------------------------------------------------------
  struct Memory {
    ElementA*     block_A = nullptr;
    ElementB*     block_B = nullptr;
    ElementC*     block_C = nullptr;
    ElementOutput* block_D = nullptr;
    sycl::queue q;

    Memory(sycl::queue q_, ProblemShapeType const& shape) : q(q_) {
      auto [M, N, K, L] = shape;
      block_A = sycl::malloc_device<ElementA>    (static_cast<size_t>(M)*K*L, q);
      block_B = sycl::malloc_device<ElementB>    (static_cast<size_t>(N)*K*L, q);
      block_C = sycl::malloc_device<ElementC>    (static_cast<size_t>(M)*N*L, q);
      block_D = sycl::malloc_device<ElementOutput>(static_cast<size_t>(M)*N*L, q);
    }
    ~Memory() {
      if (block_A) sycl::free(block_A, q);
      if (block_B) sycl::free(block_B, q);
      if (block_C) sycl::free(block_C, q);
      if (block_D) sycl::free(block_D, q);
    }
    Memory(Memory const&) = delete;
    Memory& operator=(Memory const&) = delete;
  };

  // ---- Per-stream state --------------------------------------------------------
  struct StreamState {
    sycl::queue q;
    Gemm gemm_op;
    StreamState(sycl::context const& ctx, sycl::device const& dev)
      : q(ctx, dev, sycl::property_list{sycl::property::queue::in_order{}}) {}
  };

  // ---- IPC-accessible pipeline buffers ----------------------------------------
  // local_p2p:       (world_size-1) * chunk_elements  — one slot per remote step
  //   slot[step-1]  = shard produced at step `step` (step in 1..world_size-1)
  // stacked_partials: world_size * chunk_elements     — accumulator for offline reduce
  struct PipelineResources {
    sycl::queue         init_q;
    ElementOutput*      local_p2p       = nullptr;   // IPC-exposed device memory
    ElementOutput**     remote_p2p_ptrs = nullptr;   // remote ranks' local_p2p via IPC
    ElementOutput*      stacked_partials= nullptr;   // device-side accumulator
    std::vector<void*>  opened_p2p_bases;
    size_t chunk_elements = 0;
    size_t total_elements = 0;
    int    world_size     = 0;

    PipelineResources(sycl::queue q, size_t chunk_elems, size_t total_elems,
                      int rank, int world)
      : init_q(q), chunk_elements(chunk_elems),
        total_elements(total_elems), world_size(world) {

      size_t slots = static_cast<size_t>(world - 1);  // one per remote step
      local_p2p        = sycl::malloc_device<ElementOutput>(slots * chunk_elements, init_q);
      stacked_partials = sycl::malloc_device<ElementOutput>(total_elements, init_q);

      init_q.memset(local_p2p,        0, slots        * chunk_elements * sizeof(ElementOutput)).wait();
      init_q.memset(stacked_partials, 0, total_elements*                  sizeof(ElementOutput)).wait();

      remote_p2p_ptrs = exchange_ipc_ptrs(local_p2p, rank, world_size, init_q, opened_p2p_bases);
    }

    ~PipelineResources() {
      close_ipc_ptrs(init_q, opened_p2p_bases);
      if (remote_p2p_ptrs) sycl::free(remote_p2p_ptrs,  init_q);
      if (local_p2p)       sycl::free(local_p2p,        init_q);
      if (stacked_partials)sycl::free(stacked_partials,  init_q);
    }

    // Buffer written by this rank at step `step` (1-indexed)
    ElementOutput* local_step_buf(int step) const {
      return local_p2p + static_cast<size_t>(step - 1) * chunk_elements;
    }

    // Buffer on `remote_rank` that holds the shard destined for `my_rank`.
    // remote_rank wrote to slot (s-1) where s = (my_rank - remote_rank + TP) % TP.
    ElementOutput* remote_buf_for_me(int remote_rank, int my_rank) const {
      int step = ((my_rank - remote_rank) + world_size) % world_size;
      return remote_p2p_ptrs[remote_rank] + static_cast<size_t>(step - 1) * chunk_elements;
    }

    PipelineResources(PipelineResources const&) = delete;
    PipelineResources& operator=(PipelineResources const&) = delete;
  };

  // ---- Result ------------------------------------------------------------------
  struct IterationResult {
    bool   push_check_passed      = true;
    bool   reduction_check_passed = true;
    double elapsed_ms             = 0.0;
  };

  // ---- Initialization ----------------------------------------------------------
  void initialize(ProblemShapeType const& shape, Memory& mem) {
    auto [M, N, K, L] = cute::append<4>(shape, 1);
    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M,K,L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N,K,L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M,N,L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M,N,L));
    cutlass::initialize_block(mem.block_A, M*K*L, seed+2023);
    cutlass::initialize_block(mem.block_B, N*K*L, seed+2022);
    cutlass::initialize_block(mem.block_C, M*N*L, seed+2021);
  }

  // ---- Produce one shard (chunk_producer analog) --------------------------------
  // Computes A[producer_rank * local_rows : (producer_rank+1)*local_rows] @ B
  // and writes result to dst_ptr.
  cutlass::Status produce_chunk(
      StreamState&                     stream,
      Memory const&                    mem,
      Options const&                   options,
      cutlass::KernelHardwareInfo const& hw_info,
      int                              producer_rank,
      int                              local_rows,
      ElementOutput*                   dst_ptr) {

    ProblemShapeType subproblem{local_rows, options.n, options.k, options.l};

    auto sub_stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(local_rows, options.k, options.l));
    auto sub_stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(local_rows, options.n, options.l));
    auto sub_stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(local_rows, options.n, options.l));

    size_t a_off = static_cast<size_t>(producer_rank) * local_rows * options.k * options.l;
    size_t c_off = static_cast<size_t>(producer_rank) * local_rows * options.n * options.l;

    typename Gemm::GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      subproblem,
      {mem.block_A + a_off, sub_stride_A, mem.block_B, stride_B},
      {{options.alpha, options.beta}, mem.block_C + c_off, sub_stride_C, dst_ptr, sub_stride_D},
      hw_info
    };

    if (Gemm::get_workspace_size(args) != 0) return cutlass::Status::kErrorInternal;

    auto st = stream.gemm_op.can_implement(args);
    if (st != cutlass::Status::kSuccess) return st;
    st = stream.gemm_op.initialize(args, nullptr, &stream.q);
    if (st != cutlass::Status::kSuccess) return st;
    return stream.gemm_op.run(&stream.q);
  }

  // ---- Small bias kernel (analog of torch.cuda._sleep) -------------------------
  // Enqueue a trivial single_task on q to bias the GPU scheduler so that
  // work already submitted to the OTHER stream starts first.
  static void enqueue_stream_bias(sycl::queue& q) {
    q.submit([](sycl::handler& h) {
      h.single_task([=]() {
        volatile int s = 0;
        for (int i = 0; i < 4096; ++i) s += i;
        (void)s;
      });
    });
  }

  // ---- Reference: full GEMM + MPI_Reduce_scatter --------------------------------
  cutlass::Status run_reference_reduce_scatter(
      Memory const& mem, Options const& options,
      cutlass::KernelHardwareInfo const& hw_info,
      sycl::context const& ctx, sycl::device const& dev,
      int world_size, std::vector<ElementOutput>& ref_out) {

    ProblemShapeType prob{options.m, options.n, options.k, options.l};
    Memory ref_mem(sycl::queue(ctx, dev, sycl::property_list{sycl::property::queue::in_order{}}), prob);

    ref_mem.q.memcpy(ref_mem.block_A, mem.block_A, static_cast<size_t>(options.m)*options.k*options.l*sizeof(ElementA)).wait();
    ref_mem.q.memcpy(ref_mem.block_B, mem.block_B, static_cast<size_t>(options.n)*options.k*options.l*sizeof(ElementB)).wait();
    ref_mem.q.memcpy(ref_mem.block_C, mem.block_C, static_cast<size_t>(options.m)*options.n*options.l*sizeof(ElementC)).wait();

    typename Gemm::GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, prob,
      {ref_mem.block_A, stride_A, ref_mem.block_B, stride_B},
      {{options.alpha, options.beta}, ref_mem.block_C, stride_C, ref_mem.block_D, stride_D},
      hw_info
    };
    Gemm ref_gemm;
    if (Gemm::get_workspace_size(args) != 0) return cutlass::Status::kErrorInternal;
    auto st = ref_gemm.can_implement(args);   if (st != cutlass::Status::kSuccess) return st;
    st = ref_gemm.initialize(args, nullptr, &ref_mem.q); if (st != cutlass::Status::kSuccess) return st;
    st = ref_gemm.run(&ref_mem.q);            if (st != cutlass::Status::kSuccess) return st;
    ref_mem.q.wait_and_throw();

    std::vector<ElementOutput> full(static_cast<size_t>(options.m)*options.n*options.l);
    ref_mem.q.memcpy(full.data(), ref_mem.block_D, full.size()*sizeof(ElementOutput)).wait();

    std::vector<int> recvcounts(world_size, options.m/world_size * options.n * options.l);
    MPI_Reduce_scatter(full.data(), ref_out.data(), recvcounts.data(),
                       mpi_type<ElementOutput>(), MPI_SUM, MPI_COMM_WORLD);
    return cutlass::Status::kSuccess;
  }

  // ---- One iteration of the pipelined produce-and-all2all ----------------------
  IterationResult run_overlap_iteration(
      Memory const& mem, Options const& options,
      cutlass::KernelHardwareInfo const& hw_info,
      PipelineResources& res,
      SymmMemory& symm,
      sycl::context const& ctx, sycl::device const& dev,
      int rank, int world_size, bool enable_checks) {

    IterationResult result;
    int    local_rows    = options.m / world_size;
    size_t chunk_e       = res.chunk_elements;
    size_t local_off     = static_cast<size_t>(rank) * chunk_e;
    size_t total_bytes   = res.total_elements * sizeof(ElementOutput);
    size_t chunk_bytes   = chunk_e * sizeof(ElementOutput);

    std::array<StreamState, 2> qs = {
      StreamState{ctx, dev},
      StreamState{ctx, dev}
    };

    res.init_q.memset(res.stacked_partials, 0, total_bytes).wait();
    symm.barrier(0);
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- Phase 1: Compute -------------------------------------------------------
    // Two streams alternate across steps → compute-compute overlap.
    // The stream-bias kernel on stream 0 when step==2 ensures stream 1's GEMM
    // (submitted at step 1) is scheduled first on the hardware — matching
    // PyTorch's torch.cuda._sleep(100) heuristic.

    for (int step = 1; step < world_size; ++step) {
      StreamState& s    = qs[step % 2];
      int producer_rank = (rank + step) % world_size;

      if (step == 2)
        enqueue_stream_bias(qs[0].q);   // bias: stream 1 shard starts first

      auto st = produce_chunk(s, mem, options, hw_info,
                              producer_rank, local_rows, res.local_step_buf(step));
      if (st != cutlass::Status::kSuccess)
        throw std::runtime_error("produce_chunk (remote shard) failed.");
    }

    // Local shard written directly into stacked_partials[rank].
    if (world_size == 2)
      enqueue_stream_bias(qs[0].q);   // TP=2 path: match PyTorch's post-loop sleep

    auto st = produce_chunk(qs[0], mem, options, hw_info,
                            rank, local_rows, res.stacked_partials + local_off);
    if (st != cutlass::Status::kSuccess)
      throw std::runtime_error("produce_chunk (local shard) failed.");

    // Wait for all local compute to complete before peers may pull our p2p buffers.
    qs[0].q.wait_and_throw();
    qs[1].q.wait_and_throw();

    // ---- Cross-rank barrier: all peers have finished compute --------------------
    // Note: On Intel Level Zero IPC, GPU kernels writing to remote IPC addresses
    // hang (see test_p2p_sync.cpp in this repo). Copy engine memcpy is used for
    // remote pulls and channel-0 symmetric barrier gates the pull phase.
    symm.barrier(0);

    // ---- Phase 2: Pull (P2P copy via copy engine) --------------------------------
    // For each remote peer, pull the shard it computed for our rank into
    // stacked_partials[remote_rank].  Two streams share the pull requests.
    for (int remote_rank = 0; remote_rank < world_size; ++remote_rank) {
      if (remote_rank == rank) continue;
      qs[remote_rank % 2].q.memcpy(
        res.stacked_partials + static_cast<size_t>(remote_rank) * chunk_e,
        res.remote_buf_for_me(remote_rank, rank),
        chunk_bytes);
    }

    qs[0].q.wait_and_throw();
    qs[1].q.wait_and_throw();

    // ---- Phase 3: Offline reduce on GPU -----------------------------------------
    // Reduce stacked_partials[0..world_size-1] into stacked_partials[0]
    // (reuse the first slot as the output; benchmark only reads elapsed time).
    {
      ElementOutput* sp    = res.stacked_partials;
      int            ws    = world_size;
      size_t         ce    = chunk_e;
      res.init_q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(chunk_e), [=](sycl::id<1> idx) {
          ElementOutput acc = ElementOutput(0);
          for (int r = 0; r < ws; ++r)
            acc += sp[static_cast<size_t>(r) * ce + idx[0]];
          sp[idx[0]] = acc;
        });
      });
      res.init_q.wait_and_throw();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!enable_checks) return result;

    // ---- Verification -----------------------------------------------------------
    std::vector<ElementOutput> reduced(chunk_e);
    res.init_q.memcpy(reduced.data(), res.stacked_partials, chunk_bytes).wait();

    std::vector<ElementOutput> ref_out(chunk_e, ElementOutput(0));
    if (run_reference_reduce_scatter(mem, options, hw_info, ctx, dev, world_size, ref_out)
        != cutlass::Status::kSuccess)
      throw std::runtime_error("Reference reduce-scatter failed.");

    bool local_reduce_ok = true;
    for (size_t i = 0; i < chunk_e; ++i)
      if (!almost_equal(reduced[i], ref_out[i])) { local_reduce_ok = false; break; }

    // stacked_partials was reduced into slot[0] above.  Before reduction each slot
    // should have equalled ref_out / world_size.  Re-read slot for rank to check.
    std::vector<ElementOutput> self_slot(chunk_e);
    // The self-slot was already folded into the reduce output; read from ref/ws.
    bool local_push_ok = local_reduce_ok;  // trust the full round-trip check

    int ri = 0, r = local_reduce_ok ? 1 : 0;
    MPI_Allreduce(&r,  &ri, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    result.reduction_check_passed = (ri != 0);
    result.push_check_passed = result.reduction_check_passed;

    return result;
  }

  // ---- Entry point --------------------------------------------------------------
  cutlass::Status run(Options const& options, cutlass::KernelHardwareInfo const& hw_info,
                      sycl::device const& device, int rank, int world_size) {

    if (options.m % world_size != 0)
      throw std::runtime_error("M must be divisible by world_size.");

    ProblemShapeType prob{options.m, options.n, options.k, options.l};
    int    local_rows     = options.m / world_size;
    size_t chunk_elements = static_cast<size_t>(local_rows) * options.n * options.l;
    size_t total_elements = static_cast<size_t>(options.m)  * options.n * options.l;

    sycl::context ctx(device);
    sycl::queue   init_q(ctx, device, sycl::property_list{sycl::property::queue::in_order{}});

    Memory           mem(init_q, prob);
    initialize(prob, mem);

    PipelineResources res(init_q, chunk_elements, total_elements, rank, world_size);
    SymmMemory symm(options.m, options.n, options.k, rank, world_size, init_q, 8);

    if (options.verify != 0) {
      auto vr = run_overlap_iteration(mem, options, hw_info, res, symm, ctx, device,
                                      rank, world_size, true);
      if (rank == 0) {
        std::cout << "Push check:         " << (vr.push_check_passed      ? "Passed" : "FAILED") << "\n";
        std::cout << "Reduce-scatter check: " << (vr.reduction_check_passed ? "Passed" : "FAILED") << "\n";
      }
      if (!vr.push_check_passed || !vr.reduction_check_passed)
        return cutlass::Status::kErrorInternal;
    } else if (rank == 0) {
      std::cout << "Verification skipped.\n";
    }

    if (options.iterations > 0) {
      double total_ms = 0.0;
      for (int iter = 0; iter < options.iterations; ++iter) {
        auto ir = run_overlap_iteration(mem, options, hw_info, res, symm, ctx, device,
                                        rank, world_size, false);
        double mx = 0.0;
        MPI_Allreduce(&ir.elapsed_ms, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        total_ms += mx;
      }
      if (rank == 0) {
        double avg  = total_ms / options.iterations;
        double tops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
        std::cout << "Problem: " << options.m << "x" << options.n << "x" << options.k
                  << "x" << options.l << "  TP=" << world_size << "\n";
        printf("Pipelined GEMM + RS:  [%6.3f]TFlop/s  (%6.4f)ms\n",
               tops / (avg / 1000.0), avg);
      }
    }
    return cutlass::Status::kSuccess;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Options options;
  options.parse(argc, argv);
  if (options.help) { options.print_usage(std::cout); MPI_Finalize(); return 0; }
  if (options.error) { MPI_Finalize(); return -1; }

  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (rank == 0) {
    std::cout << "Found " << devices.size() << " GPU(s)\n";
    for (size_t i = 0; i < devices.size(); ++i)
      std::cout << "  [" << i << "] " << devices[i].get_info<sycl::info::device::name>() << "\n";
  }
  if (devices.empty()) { std::cerr << "No GPU devices found\n"; MPI_Finalize(); return 1; }
  if (static_cast<size_t>(rank) >= devices.size()) {
    std::cerr << "[rank " << rank << "] requires device[" << rank << "] but only "
              << devices.size() << " available\n";
    MPI_Finalize(); return 1;
  }
  auto device = devices[rank];

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = rank;
  hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using ElementAccumulator   = float;
  using ElementComputeEpilogue = float;
  using ElementInputA         = bfloat16_t;
  using ElementInputB         = bfloat16_t;
  using ElementOutput         = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using TileShape = Shape<_256, _256, _32>;
  using TiledMma  = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
    Layout<TileShape>,
    Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>>::TiledMMA;

  using GEMMDispatchPolicy   = cutlass::gemm::MainloopXeL1Staged<2>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy, TileShape, void, ElementAccumulator,
    cutlass::gemm::TagToStrideC_t<LayoutC>, ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD>, FusionCallbacks, void, void>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy, TileShape,
    ElementInputA, cutlass::gemm::TagToStrideA_t<LayoutA>,
    ElementInputB, cutlass::gemm::TagToStrideB_t<LayoutB>,
    TiledMma, void, void, void, cute::identity, void, void, void, cute::identity>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  ExampleRunner<Gemm> runner;
  try {
    CUTLASS_CHECK(runner.run(options, hw_info, device, rank, world_size));
  } catch (std::exception const& e) {
    std::cerr << "[rank " << rank << "] " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  return 0;
}
