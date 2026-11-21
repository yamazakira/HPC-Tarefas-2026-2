#!/bin/bash
#SBATCH --job-name=MPI_hello_akira
#SBATCH --output=saida%j.out
#SBATCH --error=erro%j.err
#SBATCH --partition=amd-512
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-0:5

mpirun mpi_hello