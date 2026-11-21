#!/bin/bash
#SBATCH --job-name=MPI_OMP_trap_gpu_akira
#SBATCH --output=saida_gpu%j.out
#SBATCH --error=erro_gpu%j.err
#SBATCH --partition=gpu-4-a100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0-0:20

# --------------------------------------------------------------
# Tarefa 20 - Trapezios hibrido MPI + OpenMP + GPU (omp target)
#
# 2 nos com GPU, 1 processo por no. O programa detecta a GPU sozinho
# (omp_get_num_devices) e executa os modos cpu, cpu_sem_backup e gpu.
# Gera resultados_gpu_np<P>.json.
#
# Particao/modulos podem variar: confira com `sinfo` e `module avail`.
# Uso (dentro da pasta Tarefa20 no supercomputador):
#     sbatch script_gpu.sh
# --------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p outputs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_PROC_BIND=false

MPIRUN_OPTS=""
if mpirun --version 2>&1 | grep -qi "open mpi"; then
    MPIRUN_OPTS="--bind-to none --map-by node"
fi

echo "===== Ambiente ====="
nvidia-smi --query-gpu=name,memory.total --format=csv 2> /dev/null

echo "===== Build (com offload para GPU) ====="
# 1a opcao: GCC com nvptx (como nos slides); 2a: NVHPC (-mp=gpu)
if mpicc -O2 -fopenmp -foffload=nvptx-none -fno-stack-protector \
        main.c -o outputs/main_gpu -lm 2> outputs/build_gcc.log; then
    echo "Compilado com GCC + nvptx-none"
elif mpicc -O2 -mp=gpu main.c -o outputs/main_gpu 2> outputs/build_nvc.log; then
    echo "Compilado com NVHPC (-mp=gpu)"
else
    echo "Sem compilador com offload: compilando so para CPU (veja outputs/build_*.log)"
    mpicc -O2 -fopenmp main.c -o outputs/main_gpu -lm || exit 1
fi

for np in 1 2; do
    echo "===== ${np} processo(s) ====="
    mpirun -np ${np} ${MPIRUN_OPTS} ./outputs/main_gpu gpu
done

echo "===== Graficos ====="
if command -v python3 > /dev/null 2>&1 && python3 -c "import matplotlib" 2> /dev/null; then
    python3 plot.py
else
    echo "matplotlib indisponivel: copie os resultados_*.json e rode plot.py localmente."
fi
