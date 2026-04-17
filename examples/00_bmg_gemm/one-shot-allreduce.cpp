template <int Size>
union Vec;

template <>
union Vec<4> {
  uint16_t u16[2];
  uint32_t u32, as_scalar;
  float f32;
};

template <>
union Vec<8> {
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64, as_scalar;
  float f32[2];
};

template <>
union alignas(16) Vec<16> {
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  uint4 u128, as_scalar;
  float f32[4];
};

template <std::memory_order Sem>
__device__ __forceinline__ void put_signal(uint32_t* addr) {
//   while (cas<Sem>(addr, 0, 1) != 0)
//     ;
  *addr = 1;
}

template <std::memory_order Sem>
__device__ __forceinline__ void wait_signal(uint32_t* addr) {
//   while (cas<Sem>(addr, 1, 0) != 1)
//     ;
  *addr = 0;
}

template <bool hasPrevMemAccess, bool hasSubsequentMemAccess>
__device__ __forceinline__ void sync_remote_blocks(
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size) {
  if (threadIdx.x < world_size) {
    auto target_rank = threadIdx.x;
    if constexpr (hasPrevMemAccess) {
      put_signal<std::memory_order_release>(
          signal_pads[target_rank] + blockIdx.x * world_size + rank);
    } else {
      put_signal<std::memory_order_relaxed>(
          signal_pads[target_rank] + blockIdx.x * world_size + rank);
    }
    if constexpr (hasSubsequentMemAccess) {
      wait_signal<std::memory_order_acquire>(
          signal_pads[rank] + blockIdx.x * world_size + target_rank);
    } else {
      wait_signal<std::memory_order_relaxed>(
          signal_pads[rank] + blockIdx.x * world_size + target_rank);
    }
  }
};

template <int Alignment, typename T>
inline Vec<Alignment> ld_vec(const T* addr) {
  Vec<Alignment> vec;
  if constexpr (Alignment == 16) {
    vec.u128 = *reinterpret_cast<const uint4*>(addr);
  } else if constexpr (Alignment == 8) {
    vec.u64 = *reinterpret_cast<const uint64_t*>(addr);
  } else if constexpr (Alignment == 4) {
    vec.u32 = *reinterpret_cast<const uint32_t*>(addr);
  } else {
    static_assert(dependent_false<T>);
  }
  return vec;
}

template <int Alignment, typename T>
inline void st_vec(T* addr, const Vec<Alignment>& vec) {
  if constexpr (Alignment == 16) {
    reinterpret_cast<uint64_t*>(addr)[0] = vec.u64[0];
    reinterpret_cast<uint64_t*>(addr)[1] = vec.u64[1];
  } else if constexpr (Alignment == 8) {
    *reinterpret_cast<uint64_t*>(addr) = vec.u64;
  } else if constexpr (Alignment == 4) {
    *reinterpret_cast<uint32_t*>(addr) = vec.u32;
  } else {
    static_assert(dependent_false<T>);
  }
}

template <typename T>
inline size_t get_alignment(T ptr_or_size) {
  auto val = reinterpret_cast<uintptr_t>(ptr_or_size);
  if (val % 16 == 0) {
    return 16;
  } else if (val % 8 == 0) {
    return 8;
  } else if (val % 4 == 0) {
    return 4;
  } else if (val % 2 == 0) {
    return 2;
  } else {
    return 1;
  }
}

inline size_t get_and_verify_alignment(
    const at::Tensor& input,
    const char* op_name) {
  const size_t min_alignment = std::max(4l, input.element_size());
  const size_t ptr_alignment = at::native::memory::get_alignment(
      static_cast<size_t>(input.storage_offset() * input.element_size()));
  TORCH_CHECK(
      ptr_alignment >= min_alignment,
      op_name,
      "<",
      input.scalar_type(),
      ">: input ptr + offset must be at least ",
      min_alignment,
      "-byte aligned.");

  const size_t size_alignment = at::native::memory::get_alignment(
      static_cast<size_t>(input.numel() * input.element_size()));
  TORCH_CHECK(
      size_alignment >= min_alignment,
      op_name,
      "<",
      input.scalar_type(),
      ">: input size must be at least ",
      min_alignment,
      "-byte aligned.");
  return std::min(ptr_alignment, size_alignment);
}

// With world_size specialization: perform balanced load from all peers before
// performing reduction.
template <typename T, int alignment, int k_world_size>
__device__ inline std::enable_if_t<(k_world_size > 0), Vec<alignment>>
load_and_reduce(T** ptrs, size_t rank, size_t world_size, size_t offset) {
  Vec<alignment> vecs[k_world_size];
#pragma unroll k_world_size
  for (size_t step = 0; step < k_world_size; ++step) {
    size_t remote_rank = (rank + step) % k_world_size;
    vecs[remote_rank] =
        at::native::memory::ld_vec<alignment>(ptrs[remote_rank] + offset);
  }
  auto acc = vecs[0];
#pragma unroll k_world_size - 1
  for (size_t r = 1; r < world_size; ++r) {
    acc = add_vec<alignment, T>(acc, vecs[r]);
  }
  return acc;
}

// One-shot all-reduce is register-intensive because it stages values loaded
// from peers in registers before performing reduction. Setting the thread
// count to 512 to prevent/alleviate register spill.
constexpr size_t one_shot_all_reduce_max_num_blocks = 24;
constexpr size_t one_shot_all_reduce_max_num_threads = 512;
template <typename T, int alignment, int k_world_size>
static __launch_bounds__(one_shot_all_reduce_max_num_threads) __global__
    void one_shot_all_reduce_kernel(
        T** input_ptrs,
        T* output_ptr,
        T* input_ptr,
        size_t input_offset,
        size_t numel,
        uint32_t** signal_pads,
        size_t rank,
        size_t world_size) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);
  // copy input to shared ptr
  auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * numel_per_thread;
  auto stride = blockDim.x * gridDim.x * numel_per_thread;
  if (input_ptr) {
    for (size_t i = offset; i < numel; i += stride) {
      Vec<alignment> vec_st = at::native::memory::ld_vec<alignment>(input_ptr + i);
      at::native::memory::st_vec<alignment>(input_ptrs[rank] + input_offset + i, vec_st);
    }
  }
  // TODO make it sync with one block for no-copy case
  sync_remote_blocks<true, true>(signal_pads, rank, world_size);
  __syncthreads();

  for (size_t i = offset; i < numel; i += stride) {
    auto vec = load_and_reduce<T, alignment, k_world_size>(
        input_ptrs, rank, world_size, input_offset + i);
    at::native::memory::st_vec<alignment>(output_ptr + i, vec);
  }

  __syncthreads();
  sync_remote_blocks<true, false>(signal_pads, rank, world_size);
}

at::Tensor one_shot_all_reduce_out_impl(
    const at::Tensor& input,
    const std::optional<at::Tensor>& local_input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  auto pg = c10d::resolve_process_group(group_name);
  RECORD_PARAM_COMMS(
      static_cast<int64_t>(0),
      std::make_tuple(pg->getGroupName(), pg->getGroupDesc()),
      pg->getRank(),
      "symm_mem::one_shot_all_reduce",
      input.numel(),
      out.numel(),
      input.scalar_type(),
      std::vector<int64_t>(),
      std::vector<int64_t>(),
      -1,
      -1,
      pg->getSize());
  TORCH_CHECK(
      input.is_contiguous(), "one_shot_all_reduce: input must be contiguous.");
  TORCH_CHECK(
      out.is_contiguous(), "one_shot_all_reduce: output must be contiguous.");
  TORCH_CHECK(
      out.sizes() == input.sizes(),
      "one_shot_all_reduce: input/output size mismatch, input.sizes(): ",
      input.sizes(),
      ", output.sizes(): ",
      out.sizes());
  TORCH_CHECK(
      reduce_op == "sum",
      "one_shot_all_reduce: only sum is supported for now.");
  if (local_input.has_value()) {
    TORCH_CHECK(
        local_input->is_contiguous(),
        "one_shot_all_reduce: local input must be contiguous.");
    TORCH_CHECK(
        local_input->numel() <= input.numel(),
        "one_shot_all_reduce: local input size must be smaller than symm buffer size.");
  }
  if (input.numel() == 0) {
    TORCH_CHECK(input.scalar_type() == out.scalar_type());
    return out;
  }
  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "one_shot_all_reduce: input must be allocated with empty_strided_p2p().");

  const size_t alignment =
      get_and_verify_alignment(input, "one_shot_all_reduce");
  if (local_input.has_value()) {
    const size_t local_alignment =
        get_and_verify_alignment(*local_input, "one_shot_all_reduce");
    TORCH_CHECK(
        alignment == local_alignment,
        "one_shot_all_reduce: local input and symm buffer must have the same alignment.");
  }

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      1,
      one_shot_all_reduce_max_num_blocks,
      one_shot_all_reduce_max_num_threads,
      num_blocks,
      num_threads,
      symm_mem->get_world_size());

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      input.scalar_type(), "one_shot_all_reduce", [&]() {
        DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
          DISPATCH_WORLD_SIZES(symm_mem->get_world_size(), [&]() {
            one_shot_all_reduce_kernel<scalar_t, k_alignment, k_world_size>
                <<<num_blocks,
                   num_threads,
                   0,
                   at::cuda::getCurrentCUDAStream()>>>(
                    reinterpret_cast<scalar_t**>(
                        symm_mem->get_buffer_ptrs_dev()),
                    out.data_ptr<scalar_t>(),
                    local_input.has_value() ? local_input->data_ptr<scalar_t>()
                                            : nullptr,
                    input.storage_offset(),
                    input.numel(),
                    reinterpret_cast<uint32_t**>(
                        symm_mem->get_signal_pad_ptrs_dev()),
                    symm_mem->get_rank(),
                    symm_mem->get_world_size());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
        });
      });
  return out;
}