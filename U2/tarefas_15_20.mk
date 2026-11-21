include tarefas_11_14.mk

# ============================================================
# Unidade 2 - Tarefas 15 a 20
# Incluido pelo Makefile da U2 com:   include tarefas_15_20.mk
#
#   make tarefaNN   -> build + execucao + graficos (local)
#   make npadNN     -> submete o job no supercomputador (sbatch)
# ============================================================
NVCC ?= nvcc
MPICC ?= mpicc
MPIRUN ?= mpirun
# Bibliotecas MPI extras (ex.: Windows/MS-MPI: MPICC=gcc MPILIBS=-lmsmpi MPIRUN=mpiexec)
MPILIBS ?=
PYTHON ?= python
NP ?= 4
THREADS ?= 4

# ------------------------------------------------------------
# TAREFA 15 - Blur: sequencial x CUDA (grid multidimensional) x NPP
# ------------------------------------------------------------
IMG15 ?= imagem.jpeg

build15:
	cd Tarefa15 && mkdir -p outputs && \
	gcc -O2 blur.c -o outputs/blur -lm && \
	$(NVCC) -O2 blur_par.cu -o outputs/blur_par && \
	$(NVCC) -O2 blur_npp.cu -o outputs/blur_npp -lnppif -lnppc

run15:
	cd Tarefa15 && \
	./outputs/blur $(IMG15) outputs/blur_seq.jpg && \
	./outputs/blur_par $(IMG15) outputs/blur_cuda.jpg && \
	./outputs/blur_npp $(IMG15) outputs/blur_npp.jpg

plot15:
	cd Tarefa15 && $(PYTHON) plot.py

tarefa15: build15 run15 plot15

npad15:
	cd Tarefa15 && sbatch script_cuda.sh

# ------------------------------------------------------------
# TAREFA 16 - Soma de vetores par/impar em CUDA
# ------------------------------------------------------------
build16:
	cd Tarefa16 && mkdir -p outputs && $(NVCC) -O2 kernel.cu -o outputs/kernel

run16:
	cd Tarefa16 && ./outputs/kernel

plot16:
	cd Tarefa16 && $(PYTHON) plot.py

tarefa16: build16 run16 plot16

npad16:
	cd Tarefa16 && sbatch script_cuda.sh

# ------------------------------------------------------------
# TAREFA 17 - Ping-pong MPI (Send, Bsend, Rsend, Ssend) - 2 processos
# ------------------------------------------------------------
build17:
	cd Tarefa17 && mkdir -p outputs && \
	for m in send bsend rsend ssend; do \
		$(MPICC) -O2 pingpong_$$m.c -o outputs/pingpong_$$m $(MPILIBS) || exit 1; \
	done

run17:
	cd Tarefa17 && \
	for m in send bsend rsend ssend; do \
		$(MPIRUN) -np 2 ./outputs/pingpong_$$m || exit 1; \
	done

plot17:
	cd Tarefa17 && $(PYTHON) plot.py

tarefa17: build17 run17 plot17

npad17:
	cd Tarefa17 && sbatch script_mpi.sh

# ------------------------------------------------------------
# TAREFA 18 - Escalonador Lider-Trabalhador (onda) - NP >= 3
# ------------------------------------------------------------
build18:
	cd Tarefa18 && mkdir -p outputs && $(MPICC) -O2 main.c -o outputs/main -lm $(MPILIBS)

run18:
	cd Tarefa18 && $(MPIRUN) -np 3 ./outputs/main && \
	$(MPIRUN) -np $(NP) ./outputs/main

plot18:
	cd Tarefa18 && $(PYTHON) plot.py

tarefa18: build18 run18 plot18

npad18:
	cd Tarefa18 && sbatch script_mpi.sh

# ------------------------------------------------------------
# TAREFA 19 - y = A.x com Scatter / Bcast / Gather
# (MAX19 limita o maior tamanho de matriz nos testes locais)
# ------------------------------------------------------------
MAX19 ?= 8000

build19:
	cd Tarefa19 && mkdir -p outputs && $(MPICC) -O2 main.c -o outputs/main -lm $(MPILIBS)

run19:
	cd Tarefa19 && \
	for p in 1 2 $(NP); do \
		$(MPIRUN) -np $$p ./outputs/main $(MAX19) || exit 1; \
	done

plot19:
	cd Tarefa19 && $(PYTHON) plot.py

tarefa19: build19 run19 plot19

npad19:
	cd Tarefa19 && sbatch script_mpi.sh

# ------------------------------------------------------------
# TAREFA 20 - Trapezios hibrido MPI + OpenMP (+ GPU)
# ------------------------------------------------------------
build20:
	cd Tarefa20 && mkdir -p outputs && \
	$(MPICC) -O2 -fopenmp main.c -o outputs/main_cpu -lm $(MPILIBS)

run20:
	cd Tarefa20 && \
	for p in 1 2; do \
		OMP_NUM_THREADS=$(THREADS) $(MPIRUN) -np $$p ./outputs/main_cpu cpu || exit 1; \
	done

plot20:
	cd Tarefa20 && $(PYTHON) plot.py

tarefa20: build20 run20 plot20

npad20:
	cd Tarefa20 && sbatch script_cpu.sh

npad20gpu:
	cd Tarefa20 && sbatch script_gpu.sh

# ------------------------------------------------------------
# TODAS AS TAREFAS (11 a 20)
#   make tarefas       -> build + execucao + graficos de todas (local;
#                         11, 13 a 16 precisam de GPU NVIDIA)
#   make tarefas_cpu   -> apenas as que rodam sem GPU (12, 17 a 20)
#   make npad_todas    -> submete todos os jobs no NPAD
#   make plots         -> gera os graficos de todas que ja tem resultados
# ------------------------------------------------------------
TAREFAS = 11 12 13 14 15 16 17 18 19 20

tarefas: tarefa11 tarefa12 tarefa13 tarefa14 tarefa15 tarefa16 \
	tarefa17 tarefa18 tarefa19 tarefa20

tarefas_cpu: tarefa12 tarefa17 tarefa18 tarefa19 tarefa20

npad_todas: npad11 npad12 npad13 npad14 npad15 npad16 \
	npad17 npad18 npad19 npad20 npad20gpu

plots:
	@for t in $(TAREFAS); do \
		if ls Tarefa$$t/resultados* > /dev/null 2>&1; then \
			echo "===== Tarefa $$t ====="; \
			(cd Tarefa$$t && $(PYTHON) plot.py) || exit 1; \
		else \
			echo "===== Tarefa $$t: sem resultados, pulando ====="; \
		fi; \
	done

.PHONY: tarefas tarefas_cpu npad_todas plots build15 run15 plot15 tarefa15 npad15 build16 run16 plot16 tarefa16 npad16 \
	build17 run17 plot17 tarefa17 npad17 build18 run18 plot18 tarefa18 npad18 \
	build19 run19 plot19 tarefa19 npad19 build20 run20 plot20 tarefa20 npad20 npad20gpu
