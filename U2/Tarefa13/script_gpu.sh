#!/bin/bash
#SBATCH --job-name=OMP_GPU_onda_akira
#SBATCH --output=saida%j.out
#SBATCH --error=erro%j.err
#SBATCH --partition=gpu-4-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-0:30

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

nvidia-smi --query-gpu=name,memory.total --format=csv 2> /dev/null

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

carrega_nvhpc

for prog in main main_otimizado; do
    if command -v nvc > /dev/null 2>&1 && nvc -O2 -mp=gpu $prog.c -o outputs/$prog 2> outputs/build_nvc.log; then
        echo "$prog: NVHPC (-mp=gpu)"
    elif gcc -O2 -fopenmp -foffload=nvptx-none -fno-stack-protector $prog.c -o outputs/$prog -lm 2> outputs/build_gcc.log; then
        echo "$prog: GCC + nvptx-none"
    else
        echo "Sem compilador com offload (veja outputs/build_*.log)" >&2
        exit 1
    fi
done

./outputs/main
./outputs/main_otimizado

if command -v python3 > /dev/null 2>&1 && python3 -c "import matplotlib" 2> /dev/null; then
    python3 plot.py
fi
