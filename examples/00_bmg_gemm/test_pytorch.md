## 每个 run 的 fallback / symm ops 时间统计（单位: ms）- DLE 25.3

| run_idx | op | M | N | K | fallback_mean | fallback_median | symm_mean | symm_median |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | fused_allgather_with_matmul | 8192 | 1536 | 4096 | 3.093026 | 3.090776 | 2.812573 | 2.811276 |
| 2 | fused_allgather_with_matmul | 8192 | 7168 | 4096 | 6.990647 | 6.988280 | 5.907289 | 5.909410 |
| 3 | fused_allgather_with_matmul | 8192 | 2560 | 5120 | 4.718513 | 4.718142 | 3.654761 | 3.654456 |
| 4 | fused_allgather_with_matmul | 8192 | 12800 | 5120 | 13.553870 | 13.551304 | 12.070070 | 12.069057 |
| 5 | fused_allgather_with_matmul | 4096 | 1536 | 4096 | 1.607802 | 1.569347 | 1.472130 | 1.468246 |
| 6 | fused_allgather_with_matmul | 4096 | 7168 | 4096 | 3.601166 | 3.601260 | 3.056775 | 3.054649 |
| 7 | fused_allgather_with_matmul | 4096 | 2560 | 5120 | 2.371236 | 2.370264 | 1.873274 | 1.871961 |
| 8 | fused_allgather_with_matmul | 4096 | 12800 | 5120 | 6.801531 | 6.800599 | 6.063300 | 6.060301 |
| 9 | fused_allgather_with_matmul | 2048 | 1536 | 4096 | 1.142666 | 1.104285 | 1.505884 | 1.728766 |
| 10 | fused_allgather_with_matmul | 2048 | 7168 | 4096 | 1.953780 | 1.828970 | 1.571890 | 1.570777 |
| 11 | fused_allgather_with_matmul | 2048 | 2560 | 5120 | 1.444566 | 1.335724 | 1.898633 | 2.033889 |
| 12 | fused_allgather_with_matmul | 2048 | 12800 | 5120 | 3.416387 | 3.415217 | 3.069960 | 3.069846 |
| 13 | fused_allgather_with_matmul | 1024 | 1536 | 4096 | 0.695002 | 0.603915 | 1.694586 | 1.662622 |
| 14 | fused_allgather_with_matmul | 1024 | 7168 | 4096 | 1.222830 | 1.145781 | 1.851952 | 1.925677 |
| 15 | fused_allgather_with_matmul | 1024 | 2560 | 5120 | 0.758725 | 0.752557 | 1.583705 | 1.568502 |
| 16 | fused_allgather_with_matmul | 1024 | 12800 | 5120 | 1.799141 | 1.725152 | 1.965609 | 1.966107 |
| 17 | fused_matmul_with_reducescatter | 8192 | 4096 | 1024 | 3.195523 | 2.923381 | 3.640402 | 3.550657 |
| 18 | fused_matmul_with_reducescatter | 8192 | 4096 | 3584 | 4.865376 | 4.644224 | 5.474247 | 5.411646 |
| 19 | fused_matmul_with_reducescatter | 8192 | 5120 | 2048 | 4.677040 | 4.453930 | 5.265272 | 5.150418 |
| 20 | fused_matmul_with_reducescatter | 8192 | 5120 | 6400 | 8.372572 | 8.131182 | 8.881087 | 8.835794 |
| 21 | fused_matmul_with_reducescatter | 4096 | 4096 | 1024 | 1.910327 | 1.666100 | 2.246750 | 1.927906 |
| 22 | fused_matmul_with_reducescatter | 4096 | 4096 | 3584 | 2.654688 | 2.358557 | 3.242389 | 3.005879 |
| 23 | fused_matmul_with_reducescatter | 4096 | 5120 | 2048 | 2.471633 | 2.262033 | 2.843113 | 2.763709 |
| 24 | fused_matmul_with_reducescatter | 4096 | 5120 | 6400 | 4.303993 | 4.144023 | 4.607240 | 4.495861 |
| 25 | fused_matmul_with_reducescatter | 2048 | 4096 | 1024 | 1.080043 | 0.926179 | 1.909816 | 1.768227 |
| 26 | fused_matmul_with_reducescatter | 2048 | 4096 | 3584 | 1.943050 | 1.687472 | 1.789254 | 1.750970 |
| 27 | fused_matmul_with_reducescatter | 2048 | 5120 | 2048 | 1.710010 | 1.532908 | 1.589988 | 1.482195 |
| 28 | fused_matmul_with_reducescatter | 2048 | 5120 | 6400 | 2.341040 | 2.090016 | 2.484433 | 2.357400 |
| 29 | fused_matmul_with_reducescatter | 1024 | 4096 | 1024 | 0.908987 | 0.794612 | 1.817336 | 1.810471 |
| 30 | fused_matmul_with_reducescatter | 1024 | 4096 | 3584 | 1.286028 | 1.162980 | 1.835314 | 1.830718 |
| 31 | fused_matmul_with_reducescatter | 1024 | 5120 | 2048 | 0.943306 | 0.802054 | 1.950669 | 1.900619 |
| 32 | fused_matmul_with_reducescatter | 1024 | 5120 | 6400 | 1.555870 | 1.389414 | 1.479182 | 1.382537 |
