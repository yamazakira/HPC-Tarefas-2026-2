num ?= World
file ?= arquivo

buildrun:
	cd Tarefa$(num) && gcc $(file).c -o outputs/$(file) -Wall -pedantic && ./outputs/$(file)

buildrun1:
	cd Tarefa1 && gcc main.c -std=c11 -o outputs/main -Wall -pedantic && ./outputs/main

build1:
	cd Tarefa1 && gcc main.c -std=c11 -o outputs/main -Wall -pedantic

run1:
	cd Tarefa1 && ./outputs/main

buildrun4:
	cd Tarefa4 && gcc main.c -std=c11 -o outputs/main -Wall -pedantic -lm && ./outputs/main

run4:
	cd Tarefa4 && ./outputs/main


buildrun5:
	cd Tarefa5 && gcc main.c -std=c11 -fopenmp -o outputs/main -Wall -pedantic -lm && ./outputs/main

run5:
	cd Tarefa5 && ./outputs/main

buildrun6:
	cd Tarefa6 && gcc main.c -std=c11 -fopenmp -o outputs/main -Wall -pedantic -lm && ./outputs/main

run6:
	cd Tarefa6 && ./outputs/main

buildrun7:
	cd Tarefa7 && gcc main.c -std=c11 -fopenmp -o outputs/main -Wall -pedantic -lm && ./outputs/main

run7:
	cd Tarefa7 && ./outputs/main

buildrun7esc:
	cd Tarefa7 && gcc escalonadores.c -std=c11 -fopenmp -o outputs/escalonadores -Wall -pedantic -lm && ./outputs/escalonadores

run7:
	cd Tarefa7 && ./outputs/escalonadores

# GRAFICOS
plot1:
	python plot.py 1

all: buildrun1 build1 run1