#!/bin/bash
#SBATCH --job-name=CUDA_soma_akira
#SBATCH --output=saida%j.out
#SBATCH --error=erro%j.err
#SBATCH --partition=gpu-4-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-0:15

# --------------------------------------------------------------
# Tarefa 16 - Soma de vetores par/impar em CUDA
#
# Uso (dentro da pasta Tarefa16 no supercomputador):
#     sbatch script_cuda.sh
#
# Particao/modulo podem variar: confira com `sinfo` e `module avail cuda`.
# Para usar outro modulo: sbatch --export=ALL,CUDA_MODULE=<nome> script_cuda.sh
# --------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-.}"

if ! command -v nvcc > /dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda}" 2> /dev/null
fi

if ! command -v nvcc > /dev/null 2>&1; then
    echo "nvcc nao encontrado. Rode 'module avail cuda' e informe CUDA_MODULE." >&2
    exit 1
fi

echo "===== Ambiente ====="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version | tail -n 1

mkdir -p outputs

echo "===== Build ====="
nvcc -O2 kernel.cu -o outputs/kernel || exit 1

echo "===== Execucao ====="
./outputs/kernel

echo "===== Graficos ====="
if command -v python3 > /dev/null 2>&1 && python3 -c "import matplotlib" 2> /dev/null; then
    python3 plot.py
else
    echo "matplotlib indisponivel: copie resultados_cuda.json e rode plot.py localmente."
fi
