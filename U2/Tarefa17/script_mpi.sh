#!/bin/bash
#SBATCH --job-name=MPI_pingpong_akira
#SBATCH --output=saida%j.out
#SBATCH --error=erro%j.err
#SBATCH --partition=amd-512
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-0:15

# --------------------------------------------------------------
# Tarefa 17 - Ping-pong MPI: MPI_Send, MPI_Bsend, MPI_Rsend e MPI_Ssend
#
# Exatamente 2 processos, um em cada no (comunicacao inter-no).
# Para medir a comunicacao intra-no, troque para:
#     #SBATCH --nodes=1
#     #SBATCH --ntasks-per-node=2
#
# Uso (dentro da pasta Tarefa17 no supercomputador):
#     sbatch script_mpi.sh
# --------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs

echo "===== Build ====="
for modo in send bsend rsend ssend; do
    mpicc -O2 pingpong_${modo}.c -o outputs/pingpong_${modo} || exit 1
done

for modo in send bsend rsend ssend; do
    echo "===== MPI_${modo^} ====="
    mpirun -np 2 ./outputs/pingpong_${modo}
done

echo "===== Graficos ====="
if command -v python3 > /dev/null 2>&1 && python3 -c "import matplotlib" 2> /dev/null; then
    python3 plot.py
else
    echo "matplotlib indisponivel: copie os resultados_*.json e rode plot.py localmente."
fi
