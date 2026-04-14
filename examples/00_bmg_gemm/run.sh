LLAMA_8B_M=(
    8192 8192 8192 8192
    4096 4096 4096 4096
    2048 2048 2048 2048
    1024 1024 1024 1024
)
LLAMA_8B_N=(
    1536 7168 2560 12800
    1536 7168 2560 12800
    1536 7168 2560 12800
    1536 7168 2560 12800
)
LLAMA_8B_K=(
    4096 4096 5120 5120
    4096 4096 5120 5120
    4096 4096 5120 5120
    4096 4096 5120 5120
)

len=${#LLAMA_8B_M[@]}

for ((i=0; i<$len; i++)); do
    M=${LLAMA_8B_M[$i]}
    N=${LLAMA_8B_N[$i]}
    K=${LLAMA_8B_K[$i]}

    echo "Running fused_allgather_with_matmul with M=$M, N=$N, K=$K"
    mpirun -np 4 --prepend-rank python test_fused_allgather_matmul.py $M $N $K
    sleep 1
done


LLAMA_8B_M_1=(
    8192 8192 8192 8192
    4096 4096 4096 4096
    2048 2048 2048 2048
    1024 1024 1024 1024
)
LLAMA_8B_N_1=(
    4096 4096 5120 5120
    4096 4096 5120 5120
    4096 4096 5120 5120
    4096 4096 5120 5120
)
LLAMA_8B_K_1=(
    1024 3584 2048 6400
    1024 3584 2048 6400
    1024 3584 2048 6400
    1024 3584 2048 6400
)


for ((i=0; i<$len; i++)); do
    M=${LLAMA_8B_M_1[$i]}
    N=${LLAMA_8B_N_1[$i]}
    K=${LLAMA_8B_K_1[$i]}

    echo "Running fused_matmul_with_reducescatter with M=$M, N=$N, K=$K"
    mpirun -np 4 --prepend-rank python test_fused_matmul_reducescatter.py --M=$M --N=$N --K=$K
    sleep 1
done
