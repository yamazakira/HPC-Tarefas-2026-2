CC ?= gcc
NVCC ?= nvcc
PYTHON ?= python
OFFLOAD ?=
THREADS12 ?= 1 2 4 8

build11:
	cd Tarefa11 && mkdir -p outputs && $(CC) -O2 -fopenmp $(OFFLOAD) main.c -o outputs/main

run11:
	cd Tarefa11 && ./outputs/main

plot11:
	cd Tarefa11 && $(PYTHON) plot.py

tarefa11: build11 run11 plot11

npad11:
	cd Tarefa11 && sbatch script_gpu.sh

build12:
	cd Tarefa12 && mkdir -p outputs && $(CC) -O2 -fopenmp main.c -o outputs/main -lm

run12:
	cd Tarefa12 && THREADS="$(THREADS12)" THREADS_MASTER="1 2" bash experimentos.sh

plot12:
	cd Tarefa12 && $(PYTHON) plot.py

tarefa12: build12 run12 plot12

npad12:
	cd Tarefa12 && sbatch script_job.sh

build13:
	cd Tarefa13 && mkdir -p outputs && \
	$(CC) -O2 -fopenmp $(OFFLOAD) main.c -o outputs/main -lm && \
	$(CC) -O2 -fopenmp $(OFFLOAD) main_otimizado.c -o outputs/main_otimizado -lm

run13:
	cd Tarefa13 && ./outputs/main && ./outputs/main_otimizado

plot13:
	cd Tarefa13 && $(PYTHON) plot.py

tarefa13: build13 run13 plot13

npad13:
	cd Tarefa13 && sbatch script_gpu.sh

build14:
	cd Tarefa14 && mkdir -p outputs && \
	$(NVCC) -O2 wave.cu -o outputs/wave && \
	$(NVCC) -O2 wave_explicito.cu -o outputs/wave_explicito

run14:
	cd Tarefa14 && ./outputs/wave && ./outputs/wave_explicito

plot14:
	cd Tarefa14 && $(PYTHON) plot.py

tarefa14: build14 run14 plot14

npad14:
	cd Tarefa14 && sbatch script_cuda.sh

.PHONY: build11 run11 plot11 tarefa11 npad11 build12 run12 plot12 tarefa12 npad12 \
	build13 run13 plot13 tarefa13 npad13 build14 run14 plot14 tarefa14 npad14
