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

# GRAFICOS
plot1:
	python plot.py 1

all: buildrun1 build1 run1