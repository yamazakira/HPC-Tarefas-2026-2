#!/bin/bash
cd "$(dirname "$0")"

THREADS="${THREADS:-1 2 4 8 16 24 32}"
AFINIDADES="${AFINIDADES:-false taskset close spread master}"
THREADS_MASTER="${THREADS_MASTER:-1 2 4 8 16 24 32}"
NX_FORTE=${NX_FORTE:-2000}
NZ_FORTE=${NZ_FORTE:-1000}
NX_FRACA=${NX_FRACA:-2000}
NZ_POR_THREAD=${NZ_POR_THREAD:-100}
BIN=./outputs/main

rm -f resultados.out

executa() {
    local af=$1 t=$2 nx=$3 nz=$4 exp=$5
    if [ "$af" = "taskset" ]; then
        OMP_NUM_THREADS=$t OMP_PROC_BIND=false taskset -c 0-$((t - 1)) \
            $BIN $nx $nz 0 "$exp:$af" > /dev/null
    else
        OMP_NUM_THREADS=$t OMP_PROC_BIND=$af OMP_PLACES=cores \
            $BIN $nx $nz 0 "$exp:$af" > /dev/null
    fi
    tail -n 1 resultados.out
}

for af in $AFINIDADES; do
    lista=$THREADS
    [ "$af" = "master" ] && lista=$THREADS_MASTER
    for t in $lista; do
        executa $af $t $NX_FORTE $NZ_FORTE forte
        executa $af $t $NX_FRACA $((NZ_POR_THREAD * t)) fraca
    done
done
