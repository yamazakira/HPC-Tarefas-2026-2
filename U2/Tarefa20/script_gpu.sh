#!/bin/bash
#SBATCH --job-name=MPI_OMP_trap_gpu_akira
#SBATCH --output=saida_gpu%j.out
#SBATCH --error=erro_gpu%j.err
#SBATCH --partition=gpu-4-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=0-0:20

# --------------------------------------------------------------
# Tarefa 20 - Trapezios hibrido MPI + OpenMP + GPU (omp target)
#
# A particao gpu-4-a100 tem 1 no (4 GPUs). Aqui: 1 no, 2 processos
# MPI, 1 GPU por processo (escolhida pelo rank local no no).
# A execucao em 2 nos e feita pelo script_cpu.sh (amd-512).
# Gera resultados_gpu_np<P>.json.
#
# Uso (dentro da pasta Tarefa20 no supercomputador):
#     sbatch script_gpu.sh
# --------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_PROC_BIND=false

MPIRUN_OPTS=""
if mpirun --version 2>&1 | grep -qi "open mpi"; then
    MPIRUN_OPTS="--bind-to none --mca btl ^openib"
fi

echo "===== Ambiente ====="
nvidia-smi --query-gpu=name,memory.total --format=csv 2> /dev/null

echo "===== Build (com offload para GPU) ====="
carrega_nvhpc() {
    command -v nvc > /dev/null 2>&1 && return 0
    if ! type module > /dev/null 2>&1; then
        for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
            [ -f "$f" ] && source "$f" && break
        done
    fi
    type module > /dev/null 2>&1 || return 1
    local mod=${NVHPC_MODULE:-$(module -t avail 2>&1 | grep -v ':$' | grep -vE '^(containers|softwares)/' | grep -i 'nvhpc' | sort -V | tail -n 1)}
    [ -n "$mod" ] && echo "Carregando modulo: $mod" && module load "$mod"
    command -v nvc > /dev/null 2>&1
}

MPIRUN_BIN=$(command -v mpirun)
MPI_FLAGS="$(mpicc --showme:compile 2> /dev/null) $(mpicc --showme:link 2> /dev/null)"

if ( carrega_nvhpc && nvc -O2 -mp=gpu main.c -o outputs/main_gpu $MPI_FLAGS ) > outputs/build_nvc.log 2>&1; then
    echo "Compilado com NVHPC (-mp=gpu) + MPI do sistema"
elif mpicc -O2 -fopenmp -foffload=nvptx-none -fno-stack-protector \
        main.c -o outputs/main_gpu -lm 2> outputs/build_gcc.log; then
    echo "Compilado com GCC + nvptx-none"
else
    echo "Sem compilador com offload: compilando so para CPU (veja outputs/build_*.log)"
    mpicc -O2 -fopenmp main.c -o outputs/main_gpu -lm || exit 1
fi

for np in 1 2; do
    echo "===== ${np} processo(s) ====="
    "$MPIRUN_BIN" -np ${np} ${MPIRUN_OPTS} ./outputs/main_gpu gpu
done

echo "===== Graficos ====="
if command -v python3 > /dev/null 2>&1 && python3 -c "import matplotlib" 2> /dev/null; then
    python3 plot.py
else
    echo "matplotlib indisponivel: copie os resultados_*.json e rode plot.py localmente."
fi
