#!/bin/bash
#SBATCH --job-name=OMP_onda_akira
#SBATCH --output=saida%j.out
#SBATCH --error=erro%j.err
#SBATCH --partition=amd-512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --hint=compute_bound
#SBATCH --time=0-0:40

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs

gcc -O2 -fopenmp main.c -o outputs/main -lm || exit 1

THREADS="1 2 4 8 16 24 32" bash experimentos.sh

if command -v python3 > /dev/null 2>&1 && python3 -c "import matplotlib" 2> /dev/null; then
    python3 plot.py
fi
