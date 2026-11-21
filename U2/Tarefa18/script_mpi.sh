#!/bin/bash
#SBATCH --job-name=MPI_lider_trab_akira
#SBATCH --output=saida%j.out
#SBATCH --error=erro%j.err
#SBATCH --partition=amd-512
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=9
#SBATCH --time=0-0:20

# --------------------------------------------------------------
# Tarefa 18 - Escalonador Lider-Trabalhador (propagacao de onda)
#
# Executa com 3, 5, 9 e 17 processos (1 lider + 2, 4, 8 e 16
# trabalhadores) distribuidos em 2 nos. Cada execucao varia o numero
# de fontes sismicas (1..128) e gera resultados_np<P>.json.
#
# Uso (dentro da pasta Tarefa18 no supercomputador):
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

for np in 3 5 9 17; do
    echo "===== ${np} processos ====="
    mpirun -np ${np} ${MPIRUN_OPTS} ./outputs/main
done

echo "===== Graficos ====="
if command -v python3 > /dev/null 2>&1 && python3 -c "import matplotlib" 2> /dev/null; then
    python3 plot.py
else
    echo "matplotlib indisponivel: copie os resultados_np*.json e rode plot.py localmente."
fi
