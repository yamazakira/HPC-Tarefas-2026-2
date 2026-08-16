buildrun1:
	cd Tarefa1 && gcc main.c -std=c11 -o outputs/main -Wall -pedantic && ./outputs/main

build1:
	cd Tarefa1 && gcc main.c -std=c11 -o outputs/main -Wall -pedantic

run1:
	cd Tarefa1 && ./outputs/main

# GRAFICOS
plot1:
	python plot.py 1

all: buildrun1 build1 run1