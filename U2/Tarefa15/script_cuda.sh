#!/bin/bash
#SBATCH --job-name=CUDA_blur_akira
#SBATCH --output=saida%j.out
#SBATCH --error=erro%j.err
#SBATCH --partition=gpu-4-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-0:30

# --------------------------------------------------------------
# Tarefa 15 - Blur sequencial x CUDA (grid multidimensional) x NPP
#
# Uso (dentro da pasta Tarefa15 no supercomputador):
#     sbatch script_cuda.sh
#
# Particao/modulo podem variar: confira com `sinfo` e `module avail cuda`.
# Para usar outro modulo: sbatch --export=ALL,CUDA_MODULE=<nome> script_cuda.sh
# --------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-.}"

# Carrega o CUDA caso o nvcc nao esteja no PATH
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
gcc -O2 blur.c -o outputs/blur -lm || exit 1
nvcc -O2 blur_par.cu -o outputs/blur_par || exit 1
nvcc -O2 blur_npp.cu -o outputs/blur_npp -lnppif -lnppc || exit 1

echo "===== Sequencial ====="
./outputs/blur imagem.jpeg outputs/blur_seq.jpg

echo "===== CUDA ====="
./outputs/blur_par imagem.jpeg outputs/blur_cuda.jpg

echo "===== NPP ====="
./outputs/blur_npp imagem.jpeg outputs/blur_npp.jpg

echo "===== Graficos ====="
if command -v python3 > /dev/null 2>&1 && python3 -c "import matplotlib" 2> /dev/null; then
    python3 plot.py
else
    echo "matplotlib indisponivel: copie os resultados_*.json e rode plot.py localmente."
fi
