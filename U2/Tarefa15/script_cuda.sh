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
gcc -O2 blur.c -o outputs/blur -lm || exit 1
nvcc -O2 blur_par.cu -o outputs/blur_par || exit 1
RAIZ_CUDA=$(dirname "$(dirname "$(command -v nvcc)")")
NPP_LIB=$(find "$RAIZ_CUDA" "$RAIZ_CUDA/.." -maxdepth 6 -name 'libnppif.so*' 2> /dev/null | head -n 1)
NPP_INC=$(find "$RAIZ_CUDA" "$RAIZ_CUDA/.." -maxdepth 6 -name 'npp.h' 2> /dev/null | head -n 1)
NPP_FLAGS=""
[ -n "$NPP_INC" ] && NPP_FLAGS="$NPP_FLAGS -I$(dirname "$NPP_INC")"
[ -n "$NPP_LIB" ] && NPP_FLAGS="$NPP_FLAGS -L$(dirname "$NPP_LIB") -Xlinker -rpath=$(dirname "$NPP_LIB")"
nvcc -O2 blur_npp.cu -o outputs/blur_npp $NPP_FLAGS -lnppif -lnppc || exit 1

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
