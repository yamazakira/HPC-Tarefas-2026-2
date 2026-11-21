#!/bin/bash
#SBATCH --job-name=MPI_matvec_akira
#SBATCH --output=saida%j.out
#SBATCH --error=erro%j.err
#SBATCH --partition=amd-512
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=0-0:30

# --------------------------------------------------------------
# Tarefa 19 - y = A.x com MPI_Scatter / MPI_Bcast / MPI_Gather
#
# Executa com 1, 2, 4, 8, 16 e 32 processos (2 nos x 16 tarefas).
# Com 32 processos os dois nos sao usados; cada execucao gera
# resultados_np<P>.json com todos os tamanhos de matriz.
#
# Uso (dentro da pasta Tarefa19 no supercomputador):
#     sbatch script_mpi.sh
# --------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs

MPIRUN_OPTS=""
if mpirun --version 2>&1 | grep -qi "open mpi"; then
    MPIRUN_OPTS="--mca btl ^openib"
fi

echo "===== Build ====="
mpicc -O2 main.c -o outputs/main -lm || exit 1

for np in 1 2 4 8 16 32; do
    echo "===== ${np} processos ====="
    mpirun -np ${np} ${MPIRUN_OPTS} ./outputs/main
done

echo "===== Graficos ====="
if command -v python3 > /dev/null 2>&1 && python3 -c "import matplotlib" 2> /dev/null; then
    python3 plot.py
else
    echo "matplotlib indisponivel: copie os resultados_np*.json e rode plot.py localmente."
fi
