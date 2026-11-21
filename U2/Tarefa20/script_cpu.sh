#!/bin/bash
#SBATCH --job-name=MPI_OMP_trap_cpu_akira
#SBATCH --output=saida_cpu%j.out
#SBATCH --error=erro_cpu%j.err
#SBATCH --partition=amd-512
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=0-0:20

# --------------------------------------------------------------
# Tarefa 20 - Trapezios hibrido MPI + OpenMP (nos de CPU)
#
# 2 nos x 2 processos x 16 threads. Executa com P = 1, 2 e 4
# processos (com --map-by node, P = 2 ja usa os dois nos).
# Gera resultados_cpu_np<P>.json.
#
# Uso (dentro da pasta Tarefa20 no supercomputador):
#     sbatch script_cpu.sh
# --------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_PROC_BIND=false

# Open MPI prende cada processo a 1 nucleo por padrao: liberar para o OpenMP
MPIRUN_OPTS=""
if mpirun --version 2>&1 | grep -qi "open mpi"; then
    MPIRUN_OPTS="--bind-to none --map-by node --mca btl ^openib"
fi

echo "===== Build ====="
mpicc -O2 -fopenmp main.c -o outputs/main_cpu -lm || exit 1

for np in 1 2 4; do
    echo "===== ${np} processos x ${OMP_NUM_THREADS} threads ====="
    mpirun -np ${np} ${MPIRUN_OPTS} ./outputs/main_cpu cpu
done

echo "===== Graficos ====="
if command -v python3 > /dev/null 2>&1 && python3 -c "import matplotlib" 2> /dev/null; then
    python3 plot.py
else
    echo "matplotlib indisponivel: copie os resultados_*.json e rode plot.py localmente."
fi
