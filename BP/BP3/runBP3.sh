#!/bin/bash

function HELP {
  echo "Usage: ./runBP3.sh -m MODE"
  exit 1
}

#parse options
while getopts :m:h FLAG; do
  case $FLAG in
    m)
        mode=$OPTARG
        [[ ! $mode =~ CUDA|HIP|OpenCL|OpenMP|Serial ]] && {
            echo "Incorrect run mode provided"
            exit 1
        }
        ;;
    h)  #show help
        HELP
        ;;
    \?) #unrecognized option - show help
        HELP
        ;;
  esac
done

if [ -z $mode ]
then
    echo "No mode supplied, defaulting to HIP"
    mode=HIP
fi

# Build the code
# make -j `nproc`

echo "Running BP3..."

mpirun -np 1 BP3 -m $mode -nx 126 -ny 126 -nz 126 -p 1
mpirun -np 1 BP3 -m $mode -nx  80 -ny  80 -nz  80 -p 2
mpirun -np 1 BP3 -m $mode -nx  53 -ny  53 -nz  53 -p 3
mpirun -np 1 BP3 -m $mode -nx  40 -ny  40 -nz  40 -p 4
mpirun -np 1 BP3 -m $mode -nx  32 -ny  32 -nz  32 -p 5
mpirun -np 1 BP3 -m $mode -nx  27 -ny  27 -nz  27 -p 6
mpirun -np 1 BP3 -m $mode -nx  23 -ny  23 -nz  23 -p 7
mpirun -np 1 BP3 -m $mode -nx  20 -ny  20 -nz  20 -p 8
mpirun -np 1 BP3 -m $mode -nx  18 -ny  18 -nz  18 -p 9
mpirun -np 1 BP3 -m $mode -nx  16 -ny  16 -nz  16 -p 10
mpirun -np 1 BP3 -m $mode -nx  15 -ny  15 -nz  15 -p 11
mpirun -np 1 BP3 -m $mode -nx  14 -ny  14 -nz  14 -p 12
mpirun -np 1 BP3 -m $mode -nx  13 -ny  13 -nz  13 -p 13
mpirun -np 1 BP3 -m $mode -nx  12 -ny  12 -nz  12 -p 14

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
