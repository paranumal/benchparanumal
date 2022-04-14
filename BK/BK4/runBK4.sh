#!/bin/bash

function HELP {
  echo "Usage: ./runBK4.sh -m MODE"
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

echo "Running BK4..."

mpirun -np 1 BK4 -m $mode -nx  87 -ny  87 -nz  87 -p 1
mpirun -np 1 BK4 -m $mode -nx  56 -ny  56 -nz  56 -p 2
mpirun -np 1 BK4 -m $mode -nx  37 -ny  37 -nz  37 -p 3
mpirun -np 1 BK4 -m $mode -nx  28 -ny  28 -nz  28 -p 4
mpirun -np 1 BK4 -m $mode -nx  23 -ny  23 -nz  23 -p 5
mpirun -np 1 BK4 -m $mode -nx  19 -ny  19 -nz  19 -p 6
mpirun -np 1 BK4 -m $mode -nx  16 -ny  16 -nz  16 -p 7
mpirun -np 1 BK4 -m $mode -nx  14 -ny  14 -nz  14 -p 8
mpirun -np 1 BK4 -m $mode -nx  12 -ny  12 -nz  12 -p 9
# mpirun -np 1 BK4 -m $mode -nx  11 -ny  11 -nz  11 -p 10
# mpirun -np 1 BK4 -m $mode -nx  10 -ny  10 -nz  10 -p 11
# mpirun -np 1 BK4 -m $mode -nx  10 -ny  10 -nz  10 -p 12
# mpirun -np 1 BK4 -m $mode -nx  9 -ny  9 -nz  9 -p 13
# mpirun -np 1 BK4 -m $mode -nx  8 -ny  8 -nz  8 -p 14

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
