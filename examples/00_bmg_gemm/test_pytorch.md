## 每个 run 的 fallback / symm ops 时间统计（单位: ms）- DLE 25.3.2

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
| 17 | fused_matmul_with_reducescatter | 8192 | 4096 | 1024 | 2.986178 | 2.902276 | 2.869079 | 2.773472 |
| 18 | fused_matmul_with_reducescatter | 8192 | 4096 | 3584 | 4.676672 | 4.644146 | 3.491992 | 3.402984 |
| 19 | fused_matmul_with_reducescatter | 8192 | 5120 | 2048 | 4.790053 | 4.452058 | 3.774670 | 3.642366 |
| 20 | fused_matmul_with_reducescatter | 8192 | 5120 | 6400 | 8.175872 | 8.132254 | 6.463548 | 6.340126 |
| 21 | fused_matmul_with_reducescatter | 4096 | 4096 | 1024 | 1.760580 | 1.619020 | 2.290772 | 2.194036 |
| 22 | fused_matmul_with_reducescatter | 4096 | 4096 | 3584 | 2.378948 | 2.360592 | 2.130508 | 1.976910 |
| 23 | fused_matmul_with_reducescatter | 4096 | 5120 | 2048 | 3.041922 | 2.510326 | 2.405640 | 1.934920 |
| 24 | fused_matmul_with_reducescatter | 4096 | 5120 | 6400 | 4.322375 | 4.145284 | 3.357156 | 3.239444 |
| 25 | fused_matmul_with_reducescatter | 2048 | 4096 | 1024 | 1.205953 | 1.028560 | 2.284532 | 2.181426 |
| 26 | fused_matmul_with_reducescatter | 2048 | 4096 | 3584 | 1.828013 | 1.521286 | 2.202621 | 2.218710 |
| 27 | fused_matmul_with_reducescatter | 2048 | 5120 | 2048 | 1.945154 | 1.639248 | 2.243862 | 2.180724 |
| 28 | fused_matmul_with_reducescatter | 2048 | 5120 | 6400 | 2.577338 | 2.101736 | 1.854570 | 1.766050 |
| 29 | fused_matmul_with_reducescatter | 1024 | 4096 | 1024 | 1.127480 | 0.923026 | 2.251246 | 2.257996 |
| 30 | fused_matmul_with_reducescatter | 1024 | 4096 | 3584 | 1.195834 | 1.057758 | 2.251064 | 2.021760 |
| 31 | fused_matmul_with_reducescatter | 1024 | 5120 | 2048 | 1.135134 | 1.044238 | 2.037672 | 1.985178 |
| 32 | fused_matmul_with_reducescatter | 1024 | 5120 | 6400 | 1.276288 | 1.241812 | 2.173792 | 2.190292 |


## 每个 run 的 fallback / symm ops 时间统计（单位: ms）- DLE 25.2.2
| run_idx | op | M | N | K | fallback_mean | fallback_median | symm_mean | symm_median |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | fused_allgather_with_matmul | 8192 | 1536 | 4096 | 3.638898 | 3.638920 | 2.715757 | 2.715752 |
| 2 | fused_allgather_with_matmul | 8192 | 7168 | 4096 | 7.550689 | 7.550662 | 5.666549 | 5.666546 |
| 3 | fused_allgather_with_matmul | 8192 | 2560 | 5120 | 5.416935 | 5.417015 | 3.478150 | 3.478147 |
| 4 | fused_allgather_with_matmul | 8192 | 12800 | 5120 | 14.255692 | 14.255673 | 11.759356 | 11.759355 |
| 5 | fused_allgather_with_matmul | 4096 | 1536 | 4096 | 1.863133 | 1.867696 | 1.425363 | 1.425381 |
| 6 | fused_allgather_with_matmul | 4096 | 7168 | 4096 | 3.874199 | 3.874314 | 2.919173 | 2.919186 |
| 7 | fused_allgather_with_matmul | 4096 | 2560 | 5120 | 2.702118 | 2.702064 | 1.789501 | 1.789512 |
| 8 | fused_allgather_with_matmul | 4096 | 12800 | 5120 | 7.140684 | 7.140527 | 5.884243 | 5.884230 |
| 9 | fused_allgather_with_matmul | 2048 | 1536 | 4096 | 0.983234 | 0.983141 | 1.456223 | 1.456991 |
| 10 | fused_allgather_with_matmul | 2048 | 7168 | 4096 | 1.997938 | 2.003030 | 1.493615 | 1.493631 |
| 11 | fused_allgather_with_matmul | 2048 | 2560 | 5120 | 1.363660 | 1.363572 | 1.099654 | 1.099647 |
| 12 | fused_allgather_with_matmul | 2048 | 12800 | 5120 | 3.590295 | 3.590286 | 2.972277 | 2.972268 |
| 13 | fused_allgather_with_matmul | 1024 | 1536 | 4096 | 0.609229 | 0.609036 | 1.108932 | 1.127440 |
| 14 | fused_allgather_with_matmul | 1024 | 7168 | 4096 | 1.320503 | 1.327848 | 1.441028 | 1.442946 |
| 15 | fused_allgather_with_matmul | 1024 | 2560 | 5120 | 0.947873 | 0.948468 | 1.448910 | 1.475056 |
| 16 | fused_allgather_with_matmul | 1024 | 12800 | 5120 | 1.797537 | 1.797673 | 1.886329 | 1.886326 |
| 17 | fused_matmul_with_reducescatter | 8192 | 4096 | 1024 | 3.472108 | 3.462982 | 2.816827 | 2.815478 |
| 18 | *fused_matmul_with_reducescatter* | 8192 | 4096 | 3584 | 5.178662 | 5.184561 | 3.319928 | 3.325686 |
| 19 | fused_matmul_with_reducescatter | 8192 | 5120 | 2048 | 5.093908 | 5.069415 | 3.609579 | 3.604559 |
| 20 | fused_matmul_with_reducescatter | 8192 | 5120 | 6400 | 8.773580 | 8.773625 | 6.207282 | 6.208015 |
| 21 | fused_matmul_with_reducescatter | 4096 | 4096 | 1024 | 2.223859 | 2.291281 | 1.827606 | 1.823679 |
| 22 | fused_matmul_with_reducescatter | 4096 | 4096 | 3584 | 2.856573 | 2.849678 | 1.881230 | 1.882860 |
| 23 | fused_matmul_with_reducescatter | 4096 | 5120 | 2048 | 2.654986 | 2.646192 | 1.879691 | 1.877811 |
| 24 | fused_matmul_with_reducescatter | 4096 | 5120 | 6400 | 4.457216 | 4.447511 | 3.286830 | 3.294944 |
| 25 | fused_matmul_with_reducescatter | 2048 | 4096 | 1024 | 1.138783 | 1.145448 | 1.646324 | 1.631677 |
| 26 | fused_matmul_with_reducescatter | 2048 | 4096 | 3584 | 1.704915 | 1.738441 | 1.608015 | 1.605245 |
| 27 | fused_matmul_with_reducescatter | 2048 | 5120 | 2048 | 1.662370 | 1.632935 | 1.529436 | 1.535916 |
| 28 | fused_matmul_with_reducescatter | 2048 | 5120 | 6400 | 2.405447 | 2.399860 | 1.757331 | 1.769687 |
| 29 | fused_matmul_with_reducescatter | 1024 | 4096 | 1024 | 0.806428 | 0.804950 | 1.531067 | 1.548537 |
| 30 | fused_matmul_with_reducescatter | 1024 | 4096 | 3584 | 1.142868 | 1.141782 | 1.604227 | 1.616950 |
| 31 | fused_matmul_with_reducescatter | 1024 | 5120 | 2048 | 1.004576 | 1.037678 | 1.848027 | 1.846931 |
| 32 | fused_matmul_with_reducescatter | 1024 | 5120 | 6400 | 1.654418 | 1.644729 | 1.483491 | 1.485679 |

