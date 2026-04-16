import argparse

pcie_discount = 0.85 #0.6957
N_uni = 31.5 * pcie_discount * 1024 * 1024 * 1024
T_fp8 = 98 * 1000 * 1000 * 1000 * 1000 * 2 / 2  # for BMG
T_bf16 = 98 * 1000 * 1000 * 1000 * 1000 / 2
mem_capacity = 24
MemoryBW = 0.450 * 1000 * 1000 * 1000 * 1000
N_lat = 3 / 1000 / 1000
N_p2p_uni = N_uni

def proj_bf16_gemm(m, k, n):
    weights = k * n
    output = m * n
    macs = m * n * k
    compute_time = macs / T_bf16
    mem_time = (weights + output) * 2 / MemoryBW # Include write-back
    return max(compute_time, mem_time)

def proj_allgather(input_msg_size, intra_node_tp):
    allgather_time = 0
    size_p2p = input_msg_size
    time_p2p = (size_p2p / N_p2p_uni + N_lat) * (intra_node_tp - 1)
    allgather_time += time_p2p
    return allgather_time

def proj_reduce_scatter(input_msg_size, intra_node_tp):
    # Ring reduce-scatter: (n - 1) steps, each step transfers (1/n) of data
    size_p2p = input_msg_size / intra_node_tp
    time_p2p = (size_p2p / N_p2p_uni + N_lat) * (intra_node_tp - 1)
    reduction_time = input_msg_size / MemoryBW
    reducescatter_time = reduction_time + time_p2p
    return reducescatter_time

def proj_copy_from_remote(msg_size):
    return msg_size / N_uni + N_lat

def proj_copy_from_local(msg_size):
    return msg_size / MemoryBW  # todo: latency

def proj_reduction_time(input_msg_size, reduction_op="avg"):
    print(f"zl_debug reduction size is {input_msg_size} \n")
    return input_msg_size / MemoryBW  #todo: latency

def proj_allgather_gemm_native(m, n, k, data_type_size=2, intra_node_tp=1):
    """
    Native Allgather + GEMM (sequential, no overlap)
    Input: M, N, K, data_type_size (bytes per element)
    """
    weight_per_rank = m * k * data_type_size / intra_node_tp
    allgather_time = proj_allgather(weight_per_rank, intra_node_tp)
    gemm_time = proj_bf16_gemm(m, n, k)
    total_time = allgather_time + gemm_time
    print(
        f"[proj_allgather_gemm_native] M={m}, N={n}, K={k}, "
        f"projection={total_time * 1000:.3f}ms, "
        f"gemm={gemm_time * 1000:.3f}ms, allgather={allgather_time * 1000:.3f}ms"
    )
    return total_time

def proj_allgather_gemm(m, n, k, data_type_size=2, intra_node_tp=1):
    """
    Allgather + GEMM fusion with overlap (refer to proj_overlap pattern)
    Input: M, N, K, data_type_size (bytes per element)
    Allgather = proj_copy_from_remote + proj_copy_from_local
    """
    weight_size = m * k * data_type_size / intra_node_tp
    allgather_time = proj_copy_from_remote(weight_size)
    copy_local_time = proj_copy_from_local(weight_size)
    gemm_time = proj_bf16_gemm(m / intra_node_tp, n, k)
    
    # Overlap pattern: allgather and gemm can overlap
    overlap_time = max(allgather_time, gemm_time)
    
    if intra_node_tp == 2:
        total_time = overlap_time + gemm_time + copy_local_time
    elif intra_node_tp == 4:
        total_time = overlap_time * 3 + gemm_time + copy_local_time
    else:
        total_time = overlap_time * (intra_node_tp - 1) + gemm_time + copy_local_time

    allgather_total = allgather_time + copy_local_time
    print(
        f"[proj_allgather_gemm] M={m}, N={n}, K={k}, "
        f"projection={total_time * 1000:.3f}ms, "
        f"gemm={gemm_time * 1000:.3f}ms, allgather={allgather_total * 1000:.3f}ms"
    )
    return total_time

def proj_gemm_reducescatter_native(m, n, k, data_type_size=2, intra_node_tp=1):
    """
    Native GEMM + Reduce-Scatter (sequential, no overlap)
    Input: M, N, K, data_type_size (bytes per element)
    """
    gemm_time = proj_bf16_gemm(m, n, k)
    output_size = m * n * data_type_size
    reducescatter_time = proj_reduce_scatter(output_size, intra_node_tp)
    total_time = gemm_time + reducescatter_time
    return total_time

def proj_gemm_reducescatter(m, n, k, data_type_size=2, intra_node_tp=1):
    """
    GEMM + Reduce-Scatter fusion with overlap (refer to proj_overlap pattern)
    Input: M, N, K, data_type_size (bytes per element)
    """
    gemm_time = proj_bf16_gemm(m / intra_node_tp, n, k)
    output_size = m * n * data_type_size / intra_node_tp
    reducescatter_time = proj_copy_from_remote(output_size)
    reduction_time = proj_reduction_time(output_size) * (intra_node_tp + 1)
    
    # Overlap pattern: gemm followed by overlapped reducescatter and reduction
    if intra_node_tp == 2:
        total_time = gemm_time + max(reducescatter_time, gemm_time) + reduction_time
    elif intra_node_tp == 4:
        total_time = gemm_time + max(reducescatter_time, gemm_time) * 3 + reduction_time
    else:
        total_time = gemm_time + max(reducescatter_time, gemm_time) * (intra_node_tp - 1) + reduction_time
    
    return total_time

def parse_data_type_size(data_type):
    if data_type == "bf16":
        return 2
    if data_type == "fp16":
        return 2
    if data_type == "fp8":
        return 1
    raise ValueError(f"Unsupported data type: {data_type}")


# test code
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Projection for fused communication+GEMM patterns")
    parser.add_argument("--m", type=int, default=8192, help="GEMM M dimension")
    parser.add_argument("--n", type=int, default=4096, help="GEMM N dimension")
    parser.add_argument("--k", type=int, default=4096, help="GEMM K dimension")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp8"],
                        help="Data type")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--pattern", type=str, default="allgather_gemm",
                        choices=["allgather_gemm", "gemm_reducescatter", "both"],
                        help="Projection pattern to run")
    args = parser.parse_args()

    data_type_size = parse_data_type_size(args.dtype)
    
    time_allgather = proj_allgather(args.m * args.n * data_type_size, args.tp)
    print(f"Projected allgather time for {args.m}x{args.n} matrix with {args.tp} TP: {time_allgather * 1000:.3f} ms")
    
    print(f"Input: M={args.m}, N={args.n}, K={args.k}, dtype={args.dtype}, tp={args.tp}, pattern={args.pattern}")

    if args.pattern in ["allgather_gemm", "both"]:
        allgather_gemm_native = proj_allgather_gemm_native(args.m, args.n, args.k, data_type_size, args.tp)
        allgather_gemm_overlap = proj_allgather_gemm(args.m, args.n, args.k, data_type_size, args.tp)
        print(
            f"[allgather+gemm] native(s)={allgather_gemm_native:.8f}, "
            f"overlap(s)={allgather_gemm_overlap:.8f}"
        )

    if args.pattern in ["gemm_reducescatter", "both"]:
        gemm_reducescatter_native = proj_gemm_reducescatter_native(args.m, args.n, args.k, data_type_size, args.tp)
        gemm_reducescatter_overlap = proj_gemm_reducescatter(args.m, args.n, args.k, data_type_size, args.tp)
        print(
            f"[gemm+reducescatter] native(s)={gemm_reducescatter_native:.8f}, "
            f"overlap(s)={gemm_reducescatter_overlap:.8f}"
        )
