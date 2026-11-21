#!/bin/bash
#SBATCH --job-name=CUDA_soma_akira
#SBATCH --output=saida%j.out
#SBATCH --error=erro%j.err
#SBATCH --partition=gpu-4-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
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

carrega_cuda() {
    command -v nvcc > /dev/null 2>&1 && return 0
    if ! type module > /dev/null 2>&1; then
        for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
            [ -f "$f" ] && source "$f" && break
        done
    fi
    if type module > /dev/null 2>&1; then
        if [ -n "$CUDA_MODULE" ]; then
            module load "$CUDA_MODULE"
        else
            for padrao in 'cuda' 'nvhpc' 'nvidia'; do
                mod=$(module -t avail 2>&1 | grep -v ':$' | grep -vE '^(containers|softwares)/' | grep -i "$padrao" | sort -V | tail -n 1)
                if [ -n "$mod" ]; then
                    echo "Carregando modulo: $mod"
                    module load "$mod" && command -v nvcc > /dev/null 2>&1 && return 0
                fi
            done
        fi
    fi
    command -v nvcc > /dev/null 2>&1 && return 0
    for d in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/cuda/bin \
             /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin \
             /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin; do
        if [ -x "$d/nvcc" ]; then
            export PATH="$d:$PATH"
            return 0
        fi
    done
    return 1
}

if ! carrega_cuda; then
    echo "nvcc nao encontrado. Modulos disponiveis:" >&2
    module -t avail 2>&1 | grep -iE 'cuda|nvhpc|nvidia' >&2
    echo "Informe o modulo: sbatch --export=ALL,CUDA_MODULE=<nome> script_cuda.sh" >&2
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
