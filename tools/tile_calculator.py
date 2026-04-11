#!/usr/bin/env python3
"""
BMG GEMM Tile Shape / TiledMMA Configuration Calculator

Usage:
    python tile_calculator.py --m 2048 --n 1536 --k 4096
    python tile_calculator.py --m 2048 --n 1536 --k 4096 --xe_cores 20 --top 10
    python tile_calculator.py --m 2048 --n 1536 --k 4096 --dtype bf16 --acc_dtype fp32

Outputs ranked configurations with C++ code snippets ready to paste.
"""

import argparse
import itertools
from dataclasses import dataclass


# ─── Hardware constants ───────────────────────────────────────────────────────

BMG_XE_CORES_DEFAULT = 20       # Arc B580
BMG_SLM_PER_CORE_KB = 128      # 128 KB SLM per Xe-core
BMG_PEAK_TFLOPS_BF16 = 93.0    # BF16 peak TFLOPS (theoretical)

# ─── DPAS atom shapes ────────────────────────────────────────────────────────

DPAS_ATOMS = {
    # (M, N, K_per_atom) for each dtype
    "bf16": (8, 16, 16),   # XE_DPAS_TT<8, float, bfloat16_t>: Shape_MNK = (8, 16, 16)
    "fp16": (8, 16, 16),   # XE_8x16x16_F32F16F16F32_TT
    "fp8":  (8, 16, 32),   # 8-bit: K doubles
    "int8": (8, 16, 32),   # XE_8x16x32_S32S8S8S32_TT
}

DTYPE_BYTES = {
    "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "fp32": 4,
}

MMA_ATOM_NAMES = {
    "bf16": "XE_DPAS_TT<8, float, cute::bfloat16_t>",
    "fp16": "XE_8x16x16_F32F16F16F32_TT",
    "fp8":  "XE_8x16x16_F32F16F16F32_TT",
    "int8": "XE_8x16x32_S32S8S8S32_TT",
}

ACC_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "int32": 4}


# ─── Valid search space ──────────────────────────────────────────────────────

TILE_M_OPTIONS = [64, 128, 256, 512]
TILE_N_OPTIONS = [64, 128, 256, 512]
TILE_K_OPTIONS = [16, 32, 64]
PIPELINE_STAGES_OPTIONS = [2, 3]

# SG layout options: (sg_m, sg_n) — total SGs = sg_m * sg_n
SG_LAYOUTS = [
    (1, 4),   # 4 SGs
    (1, 8),   # 8 SGs
    (2, 4),   # 8 SGs
    (4, 4),   # 16 SGs
    (4, 8),   # 32 SGs
    (8, 4),   # 32 SGs
    (8, 8),   # 64 SGs — rare, may exceed register budget
]


@dataclass
class Config:
    tile_m: int
    tile_n: int
    tile_k: int
    sg_m: int
    sg_n: int
    stages: int
    # derived
    atom_m: int
    atom_n: int
    atom_k: int
    iters_m: int
    iters_n: int
    iters_k: int
    total_sgs: int
    tiles_m: int
    tiles_n: int
    total_tiles: int
    waves: float
    last_wave_util: float
    acc_regs_per_sg: int     # accumulator elements per SG
    slm_kb: float            # estimated SLM usage (double-buffered A+B per tile)
    arith_intensity: float   # FLOPs / bytes loaded
    score: float = 0.0

    def sg_shape_str(self):
        return f"Shape<_{self.sg_m}, _{self.sg_n}, _1>"

    def sg_stride_str(self):
        return f"Stride<_{self.sg_n}, _1, _0>"

    def tile_shape_str(self):
        return f"Shape<_{self.tile_m}, _{self.tile_n}, _{self.tile_k}>"


def evaluate_config(M, N, K, tile_m, tile_n, tile_k, sg_m, sg_n, stages,
                    atom_m, atom_n, atom_k, xe_cores, dtype, acc_dtype):
    """Evaluate a single configuration. Returns Config or None if invalid."""

    total_sgs = sg_m * sg_n

    # Hardware coverage
    hw_m = atom_m * sg_m
    hw_n = atom_n * sg_n
    hw_k = atom_k  # K dimension: no SG tiling

    # Check divisibility: tile must be integer multiple of hardware coverage
    if tile_m % hw_m != 0 or tile_n % hw_n != 0 or tile_k % hw_k != 0:
        return None

    iters_m = tile_m // hw_m
    iters_n = tile_n // hw_n
    iters_k = tile_k // hw_k

    # Check problem divisibility (with padding tolerance)
    # Allow non-exact divisibility — padding is handled by the kernel
    tiles_m = (M + tile_m - 1) // tile_m
    tiles_n = (N + tile_n - 1) // tile_n
    total_tiles = tiles_m * tiles_n

    if total_tiles == 0:
        return None

    # Wave analysis
    waves = total_tiles / xe_cores
    last_wave_tiles = total_tiles % xe_cores
    if last_wave_tiles == 0:
        last_wave_util = 1.0
    else:
        last_wave_util = last_wave_tiles / xe_cores

    # Accumulator register pressure per SG
    # Each SG handles (iters_m * atom_m) × (iters_n * atom_n) output elements
    sg_out_m = iters_m * atom_m
    sg_out_n = iters_n * atom_n
    acc_elements = sg_out_m * sg_out_n
    acc_bytes = acc_elements * ACC_BYTES.get(acc_dtype, 4)
    acc_regs = acc_elements  # in elements (for display)

    # SLM estimate: double-buffered A tile + B tile per workgroup
    elem_bytes = DTYPE_BYTES.get(dtype, 2)
    a_tile_bytes = tile_m * tile_k * elem_bytes
    b_tile_bytes = tile_n * tile_k * elem_bytes
    slm_bytes = (a_tile_bytes + b_tile_bytes) * stages
    slm_kb = slm_bytes / 1024

    # Reject if SLM exceeds hardware limit
    if slm_kb > BMG_SLM_PER_CORE_KB:
        return None

    # Reject if too many accumulator registers (heuristic: >4096 elements)
    if acc_elements > 4096:
        return None

    # Arithmetic intensity: FLOPs per byte of global memory loaded
    flops_per_tile = 2 * tile_m * tile_n * K  # full K
    bytes_per_tile = (tile_m * K + tile_n * K) * elem_bytes  # A + B (each loaded once ideally)
    arith_intensity = flops_per_tile / bytes_per_tile if bytes_per_tile > 0 else 0

    return Config(
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        sg_m=sg_m, sg_n=sg_n, stages=stages,
        atom_m=atom_m, atom_n=atom_n, atom_k=atom_k,
        iters_m=iters_m, iters_n=iters_n, iters_k=iters_k,
        total_sgs=total_sgs,
        tiles_m=tiles_m, tiles_n=tiles_n, total_tiles=total_tiles,
        waves=waves, last_wave_util=last_wave_util,
        acc_regs_per_sg=acc_regs,
        slm_kb=slm_kb,
        arith_intensity=arith_intensity,
    )


def score_config(cfg: Config):
    """Heuristic scoring (higher = better). Weights tuned for BMG."""
    s = 0.0

    # 1. Wave utilization (0-30 pts)
    s += cfg.last_wave_util * 30

    # 2. Arithmetic intensity (0-30 pts, log scale)
    import math
    s += min(30, math.log2(max(cfg.arith_intensity, 1)) * 5)

    # 3. Prefer fewer K iterations (less loop overhead) (0-15 pts)
    k_iters = cfg.tile_k // cfg.atom_k
    s += min(15, k_iters * 5)  # K=64 → 4 iters → 15 pts; K=32 → 2 iters → 10 pts

    # 4. Sweet-spot SG count: 32 is ideal for BMG (0-10 pts)
    sg_score = {4: 2, 8: 5, 16: 7, 32: 10, 64: 6}
    s += sg_score.get(cfg.total_sgs, 3)

    # 5. Deeper pipeline (0-10 pts)
    s += (cfg.stages - 1) * 5

    # 6. Penalty: too many waves (diminishing returns from parallelism)
    if cfg.waves > 10:
        s -= (cfg.waves - 10) * 0.5

    # 7. Penalty: very high register pressure
    if cfg.acc_regs_per_sg > 2048:
        s -= (cfg.acc_regs_per_sg - 2048) / 256

    cfg.score = s
    return s


def generate_cpp(cfg: Config, dtype: str):
    """Generate C++ code snippet for a configuration."""
    atom_name = MMA_ATOM_NAMES.get(dtype, MMA_ATOM_NAMES["bf16"])
    return f"""\
using TileShape = {cfg.tile_shape_str()};
using TiledMma = typename TiledMMAHelper<MMA_Atom<{atom_name}>,
    Layout<TileShape>, Layout<{cfg.sg_shape_str()}, {cfg.sg_stride_str()}>>::TiledMMA;
constexpr int PipelineStages = {cfg.stages};
using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;"""


def main():
    parser = argparse.ArgumentParser(description="BMG GEMM Tile Configuration Calculator")
    parser.add_argument("--m", type=int, required=True, help="M dimension")
    parser.add_argument("--n", type=int, required=True, help="N dimension")
    parser.add_argument("--k", type=int, required=True, help="K dimension")
    parser.add_argument("--xe_cores", type=int, default=BMG_XE_CORES_DEFAULT,
                        help=f"Number of Xe-cores (default: {BMG_XE_CORES_DEFAULT})")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=list(DPAS_ATOMS.keys()), help="Input data type")
    parser.add_argument("--acc_dtype", type=str, default="fp32",
                        choices=list(ACC_BYTES.keys()), help="Accumulator data type")
    parser.add_argument("--top", type=int, default=10, help="Show top N configurations")
    parser.add_argument("--code", action="store_true", help="Print C++ code for top configs")
    args = parser.parse_args()

    atom_m, atom_n, atom_k = DPAS_ATOMS[args.dtype]

    print(f"{'='*80}")
    print(f"BMG GEMM Tile Calculator")
    print(f"{'='*80}")
    print(f"Problem:    M={args.m}, N={args.n}, K={args.k}")
    print(f"Hardware:   {args.xe_cores} Xe-cores, {BMG_SLM_PER_CORE_KB} KB SLM/core")
    print(f"DPAS Atom:  {atom_m}×{atom_n}×{atom_k} ({args.dtype} → {args.acc_dtype})")
    print(f"{'='*80}\n")

    configs = []
    for tile_m, tile_n, tile_k, (sg_m, sg_n), stages in itertools.product(
            TILE_M_OPTIONS, TILE_N_OPTIONS, TILE_K_OPTIONS, SG_LAYOUTS, PIPELINE_STAGES_OPTIONS):
        cfg = evaluate_config(
            args.m, args.n, args.k,
            tile_m, tile_n, tile_k,
            sg_m, sg_n, stages,
            atom_m, atom_n, atom_k,
            args.xe_cores, args.dtype, args.acc_dtype,
        )
        if cfg is not None:
            score_config(cfg)
            configs.append(cfg)

    # Sort by score descending
    configs.sort(key=lambda c: c.score, reverse=True)

    # Deduplicate: keep best per (tile_m, tile_n, tile_k, sg_m, sg_n, stages)
    seen = set()
    unique = []
    for c in configs:
        key = (c.tile_m, c.tile_n, c.tile_k, c.sg_m, c.sg_n, c.stages)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    configs = unique[:args.top]

    # Print table
    header = (f"{'#':>3} {'Score':>6} {'TileShape':>16} {'SG':>6} {'St':>2} "
              f"{'Tiles':>6} {'Waves':>6} {'LastW%':>6} "
              f"{'AccReg':>6} {'SLM_KB':>7} {'ArithI':>7}")
    print(header)
    print("-" * len(header))

    for i, c in enumerate(configs, 1):
        tile_str = f"{c.tile_m}x{c.tile_n}x{c.tile_k}"
        sg_str = f"{c.sg_m}x{c.sg_n}"
        print(f"{i:3d} {c.score:6.1f} {tile_str:>16} {sg_str:>6} {c.stages:2d} "
              f"{c.total_tiles:6d} {c.waves:6.1f} {c.last_wave_util:6.0%} "
              f"{c.acc_regs_per_sg:6d} {c.slm_kb:7.1f} {c.arith_intensity:7.1f}")

    if args.code:
        print(f"\n{'='*80}")
        print("C++ Code Snippets (paste into your GEMM configuration)")
        print(f"{'='*80}")
        for i, c in enumerate(configs, 1):
            print(f"\n// ── Config #{i} (score={c.score:.1f}) ──")
            print(generate_cpp(c, args.dtype))

    # Performance estimates
    print(f"\n{'='*80}")
    print("Performance Estimates (theoretical)")
    print(f"{'='*80}")
    total_flops = 2.0 * args.m * args.n * args.k
    peak_us = total_flops / (BMG_PEAK_TFLOPS_BF16 * 1e6)
    print(f"Total FLOPs:       {total_flops:.2e}")
    print(f"Peak time (@{BMG_PEAK_TFLOPS_BF16} TFLOPS): {peak_us:.0f} us")
    for i, c in enumerate(configs[:3], 1):
        eff = c.last_wave_util * 0.85  # rough wave-util × pipeline efficiency
        est_us = peak_us / eff
        print(f"Config #{i} est:     ~{est_us:.0f} us ({eff:.0%} efficiency)")


if __name__ == "__main__":
    main()
