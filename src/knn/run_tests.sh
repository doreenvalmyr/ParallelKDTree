#!/bin/bash

cores=(2 4 8 16 32 64 128)
for i in "${cores[@]}";
do
    echo "Running on $i cores"
    export NUM_PROCS=$i
    make run-mpi
done
