===== CASE 1/32 pattern=allgather_gemm M=1024 N=1536 K=4096 =====
CMD: python3 projection.py --m 1024 --n 1536 --k 4096 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=1024, N=1536, K=4096, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=1024, N=1536, K=4096, projection=0.359ms, gemm=0.131ms, allgather=0.228ms
[proj_allgather_gemm] M=1024, N=1536, K=4096, projection=0.265ms, gemm=0.033ms, allgather=0.081ms
[allgather+gemm] native(s)=0.00035932, overlap(s)=0.00026537
--- STDERR ---

RETURN_CODE: 0

===== CASE 2/32 pattern=allgather_gemm M=2048 N=1536 K=4096 =====
CMD: python3 projection.py --m 2048 --n 1536 --k 4096 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=2048, N=1536, K=4096, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=2048, N=1536, K=4096, projection=0.710ms, gemm=0.263ms, allgather=0.447ms
[proj_allgather_gemm] M=2048, N=1536, K=4096, projection=0.522ms, gemm=0.066ms, allgather=0.158ms
[allgather+gemm] native(s)=0.00070963, overlap(s)=0.00052174
--- STDERR ---

RETURN_CODE: 0

===== CASE 3/32 pattern=allgather_gemm M=4096 N=1536 K=4096 =====
CMD: python3 projection.py --m 4096 --n 1536 --k 4096 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=4096, N=1536, K=4096, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=4096, N=1536, K=4096, projection=1.410ms, gemm=0.526ms, allgather=0.884ms
[proj_allgather_gemm] M=4096, N=1536, K=4096, projection=1.034ms, gemm=0.131ms, allgather=0.313ms
[allgather+gemm] native(s)=0.00141026, overlap(s)=0.00103447
--- STDERR ---

RETURN_CODE: 0

===== CASE 4/32 pattern=allgather_gemm M=8192 N=1536 K=4096 =====
CMD: python3 projection.py --m 8192 --n 1536 --k 4096 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=8192, N=1536, K=4096, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=8192, N=1536, K=4096, projection=2.812ms, gemm=1.052ms, allgather=1.760ms
[proj_allgather_gemm] M=8192, N=1536, K=4096, projection=2.060ms, gemm=0.263ms, allgather=0.624ms
[allgather+gemm] native(s)=0.00281153, overlap(s)=0.00205994
--- STDERR ---

RETURN_CODE: 0

===== CASE 5/32 pattern=allgather_gemm M=1024 N=2560 K=5120 =====
CMD: python3 projection.py --m 1024 --n 2560 --k 5120 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=1024, N=2560, K=5120, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=1024, N=2560, K=5120, projection=0.556ms, gemm=0.274ms, allgather=0.283ms
[proj_allgather_gemm] M=1024, N=2560, K=5120, projection=0.357ms, gemm=0.068ms, allgather=0.100ms
[allgather+gemm] native(s)=0.00055646, overlap(s)=0.00035685
--- STDERR ---

RETURN_CODE: 0

===== CASE 6/32 pattern=allgather_gemm M=2048 N=2560 K=5120 =====
CMD: python3 projection.py --m 2048 --n 2560 --k 5120 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=2048, N=2560, K=5120, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=2048, N=2560, K=5120, projection=1.104ms, gemm=0.548ms, allgather=0.556ms
[proj_allgather_gemm] M=2048, N=2560, K=5120, projection=0.705ms, gemm=0.137ms, allgather=0.197ms
[allgather+gemm] native(s)=0.00110392, overlap(s)=0.00070470
--- STDERR ---

RETURN_CODE: 0

===== CASE 7/32 pattern=allgather_gemm M=4096 N=2560 K=5120 =====
CMD: python3 projection.py --m 4096 --n 2560 --k 5120 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=4096, N=2560, K=5120, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=4096, N=2560, K=5120, projection=2.199ms, gemm=1.096ms, allgather=1.103ms
[proj_allgather_gemm] M=4096, N=2560, K=5120, projection=1.400ms, gemm=0.274ms, allgather=0.391ms
[allgather+gemm] native(s)=0.00219884, overlap(s)=0.00140040
--- STDERR ---

RETURN_CODE: 0

===== CASE 8/32 pattern=allgather_gemm M=8192 N=2560 K=5120 =====
CMD: python3 projection.py --m 8192 --n 2560 --k 5120 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=8192, N=2560, K=5120, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=8192, N=2560, K=5120, projection=4.389ms, gemm=2.191ms, allgather=2.197ms
[proj_allgather_gemm] M=8192, N=2560, K=5120, projection=2.792ms, gemm=0.548ms, allgather=0.779ms
[allgather+gemm] native(s)=0.00438869, overlap(s)=0.00279181
--- STDERR ---

RETURN_CODE: 0

===== CASE 9/32 pattern=allgather_gemm M=1024 N=7168 K=4096 =====
CMD: python3 projection.py --m 1024 --n 7168 --k 4096 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=1024, N=7168, K=4096, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=1024, N=7168, K=4096, projection=0.841ms, gemm=0.614ms, allgather=0.228ms
[proj_allgather_gemm] M=1024, N=7168, K=4096, projection=0.618ms, gemm=0.153ms, allgather=0.081ms
[allgather+gemm] native(s)=0.00084140, overlap(s)=0.00061823
--- STDERR ---

RETURN_CODE: 0

===== CASE 10/32 pattern=allgather_gemm M=2048 N=7168 K=4096 =====
CMD: python3 projection.py --m 2048 --n 7168 --k 4096 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=2048, N=7168, K=4096, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=2048, N=7168, K=4096, projection=1.674ms, gemm=1.227ms, allgather=0.447ms
[proj_allgather_gemm] M=2048, N=7168, K=4096, projection=1.236ms, gemm=0.307ms, allgather=0.158ms
[allgather+gemm] native(s)=0.00167381, overlap(s)=0.00123645
--- STDERR ---

RETURN_CODE: 0

===== CASE 11/32 pattern=allgather_gemm M=4096 N=7168 K=4096 =====
CMD: python3 projection.py --m 4096 --n 7168 --k 4096 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=4096, N=7168, K=4096, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=4096, N=7168, K=4096, projection=3.339ms, gemm=2.454ms, allgather=0.884ms
[proj_allgather_gemm] M=4096, N=7168, K=4096, projection=2.473ms, gemm=0.614ms, allgather=0.313ms
[allgather+gemm] native(s)=0.00333862, overlap(s)=0.00247291
--- STDERR ---

RETURN_CODE: 0

===== CASE 12/32 pattern=allgather_gemm M=8192 N=7168 K=4096 =====
CMD: python3 projection.py --m 8192 --n 7168 --k 4096 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=8192, N=7168, K=4096, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=8192, N=7168, K=4096, projection=6.668ms, gemm=4.909ms, allgather=1.760ms
[proj_allgather_gemm] M=8192, N=7168, K=4096, projection=4.946ms, gemm=1.227ms, allgather=0.624ms
[allgather+gemm] native(s)=0.00666823, overlap(s)=0.00494582
--- STDERR ---

RETURN_CODE: 0

===== CASE 13/32 pattern=allgather_gemm M=1024 N=12800 K=5120 =====
CMD: python3 projection.py --m 1024 --n 12800 --k 5120 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=1024, N=12800, K=5120, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=1024, N=12800, K=5120, projection=1.652ms, gemm=1.370ms, allgather=0.283ms
[proj_allgather_gemm] M=1024, N=12800, K=5120, projection=1.375ms, gemm=0.342ms, allgather=0.100ms
[allgather+gemm] native(s)=0.00165212, overlap(s)=0.00137539
--- STDERR ---

RETURN_CODE: 0

===== CASE 14/32 pattern=allgather_gemm M=2048 N=12800 K=5120 =====
CMD: python3 projection.py --m 2048 --n 12800 --k 5120 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=2048, N=12800, K=5120, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=2048, N=12800, K=5120, projection=3.295ms, gemm=2.739ms, allgather=0.556ms
[proj_allgather_gemm] M=2048, N=12800, K=5120, projection=2.751ms, gemm=0.685ms, allgather=0.197ms
[allgather+gemm] native(s)=0.00329523, overlap(s)=0.00275079
--- STDERR ---

RETURN_CODE: 0

===== CASE 15/32 pattern=allgather_gemm M=4096 N=12800 K=5120 =====
CMD: python3 projection.py --m 4096 --n 12800 --k 5120 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=4096, N=12800, K=5120, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=4096, N=12800, K=5120, projection=6.581ms, gemm=5.478ms, allgather=1.103ms
[proj_allgather_gemm] M=4096, N=12800, K=5120, projection=5.502ms, gemm=1.370ms, allgather=0.391ms
[allgather+gemm] native(s)=0.00658146, overlap(s)=0.00550158
--- STDERR ---

RETURN_CODE: 0

===== CASE 16/32 pattern=allgather_gemm M=8192 N=12800 K=5120 =====
CMD: python3 projection.py --m 8192 --n 12800 --k 5120 --pattern allgather_gemm --tp 4 --dtype bf16
--- STDOUT ---
Input: M=8192, N=12800, K=5120, dtype=bf16, tp=4, pattern=allgather_gemm
[proj_allgather_gemm_native] M=8192, N=12800, K=5120, projection=13.154ms, gemm=10.957ms, allgather=2.197ms
[proj_allgather_gemm] M=8192, N=12800, K=5120, projection=11.003ms, gemm=2.739ms, allgather=0.779ms
[allgather+gemm] native(s)=0.01315392, overlap(s)=0.01100315
--- STDERR ---

RETURN_CODE: 0

===== CASE 17/32 pattern=gemm_reducescatter M=1024 N=4096 K=1024 =====
CMD: python3 projection.py --m 1024 --n 4096 --k 1024 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=1024, N=4096, K=1024, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 2097152.0

[gemm+reducescatter] native(s)=0.00033413, overlap(s)=0.00027305
--- STDERR ---

RETURN_CODE: 0

===== CASE 18/32 pattern=gemm_reducescatter M=2048 N=4096 K=1024 =====
CMD: python3 projection.py --m 2048 --n 4096 --k 1024 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=2048, N=4096, K=1024, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 4194304.0

[gemm+reducescatter] native(s)=0.00065926, overlap(s)=0.00053710
--- STDERR ---

RETURN_CODE: 0

===== CASE 19/32 pattern=gemm_reducescatter M=4096 N=4096 K=1024 =====
CMD: python3 projection.py --m 4096 --n 4096 --k 1024 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=4096, N=4096, K=1024, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 8388608.0

[gemm+reducescatter] native(s)=0.00130953, overlap(s)=0.00106521
--- STDERR ---

RETURN_CODE: 0

===== CASE 20/32 pattern=gemm_reducescatter M=8192 N=4096 K=1024 =====
CMD: python3 projection.py --m 8192 --n 4096 --k 1024 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=8192, N=4096, K=1024, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 16777216.0

[gemm+reducescatter] native(s)=0.00261005, overlap(s)=0.00212142
--- STDERR ---

RETURN_CODE: 0

===== CASE 21/32 pattern=gemm_reducescatter M=1024 N=4096 K=3584 =====
CMD: python3 projection.py --m 1024 --n 4096 --k 3584 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=1024, N=4096, K=3584, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 2097152.0

[gemm+reducescatter] native(s)=0.00055326, overlap(s)=0.00033009
--- STDERR ---

RETURN_CODE: 0

===== CASE 22/32 pattern=gemm_reducescatter M=2048 N=4096 K=3584 =====
CMD: python3 projection.py --m 2048 --n 4096 --k 3584 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=2048, N=4096, K=3584, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 4194304.0

[gemm+reducescatter] native(s)=0.00109752, overlap(s)=0.00066017
--- STDERR ---

RETURN_CODE: 0

===== CASE 23/32 pattern=gemm_reducescatter M=4096 N=4096 K=3584 =====
CMD: python3 projection.py --m 4096 --n 4096 --k 3584 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=4096, N=4096, K=3584, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 8388608.0

[gemm+reducescatter] native(s)=0.00218605, overlap(s)=0.00132034
--- STDERR ---

RETURN_CODE: 0

===== CASE 24/32 pattern=gemm_reducescatter M=8192 N=4096 K=3584 =====
CMD: python3 projection.py --m 8192 --n 4096 --k 3584 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=8192, N=4096, K=3584, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 16777216.0

[gemm+reducescatter] native(s)=0.00436310, overlap(s)=0.00264068
--- STDERR ---

RETURN_CODE: 0

===== CASE 25/32 pattern=gemm_reducescatter M=1024 N=5120 K=2048 =====
CMD: python3 projection.py --m 1024 --n 5120 --k 2048 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=1024, N=5120, K=2048, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 2621440.0

[gemm+reducescatter] native(s)=0.00052498, overlap(s)=0.00036646
--- STDERR ---

RETURN_CODE: 0

===== CASE 26/32 pattern=gemm_reducescatter M=2048 N=5120 K=2048 =====
CMD: python3 projection.py --m 2048 --n 5120 --k 2048 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=2048, N=5120, K=2048, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 5242880.0

[gemm+reducescatter] native(s)=0.00104096, overlap(s)=0.00072391
--- STDERR ---

RETURN_CODE: 0

===== CASE 27/32 pattern=gemm_reducescatter M=4096 N=5120 K=2048 =====
CMD: python3 projection.py --m 4096 --n 5120 --k 2048 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=4096, N=5120, K=2048, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 10485760.0

[gemm+reducescatter] native(s)=0.00207292, overlap(s)=0.00143883
--- STDERR ---

RETURN_CODE: 0

===== CASE 28/32 pattern=gemm_reducescatter M=8192 N=5120 K=2048 =====
CMD: python3 projection.py --m 8192 --n 5120 --k 2048 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=8192, N=5120, K=2048, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 20971520.0

[gemm+reducescatter] native(s)=0.00413684, overlap(s)=0.00286865
--- STDERR ---

RETURN_CODE: 0

===== CASE 29/32 pattern=gemm_reducescatter M=1024 N=5120 K=6400 =====
CMD: python3 projection.py --m 1024 --n 5120 --k 6400 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=1024, N=5120, K=6400, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 2621440.0

[gemm+reducescatter] native(s)=0.00099063, overlap(s)=0.00071391
--- STDERR ---

RETURN_CODE: 0

===== CASE 30/32 pattern=gemm_reducescatter M=2048 N=5120 K=6400 =====
CMD: python3 projection.py --m 2048 --n 5120 --k 6400 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=2048, N=5120, K=6400, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 5242880.0

[gemm+reducescatter] native(s)=0.00197227, overlap(s)=0.00142782
--- STDERR ---

RETURN_CODE: 0

===== CASE 31/32 pattern=gemm_reducescatter M=4096 N=5120 K=6400 =====
CMD: python3 projection.py --m 4096 --n 5120 --k 6400 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=4096, N=5120, K=6400, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 10485760.0

[gemm+reducescatter] native(s)=0.00393553, overlap(s)=0.00285565
--- STDERR ---

RETURN_CODE: 0

===== CASE 32/32 pattern=gemm_reducescatter M=8192 N=5120 K=6400 =====
CMD: python3 projection.py --m 8192 --n 5120 --k 6400 --pattern gemm_reducescatter --tp 4 --dtype bf16
--- STDOUT ---
Input: M=8192, N=5120, K=6400, dtype=bf16, tp=4, pattern=gemm_reducescatter
zl_debug reduction size is 20971520.0

[gemm+reducescatter] native(s)=0.00786206, overlap(s)=0.00571129
--- STDERR ---

RETURN_CODE: 0
