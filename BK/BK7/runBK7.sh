#!/bin/bash

mpi="mpirun -np 1 "
exe=./BK6

function HELP {
  echo "Usage: ./runBK6.sh -m MODE -e ELEMENT -n NDOFS"
  exit 1
}

#defaults
element=Hex
ndofs=4000000
affine=false

#parse options
while getopts :m:e:n:ah FLAG; do
  case $FLAG in
    m)
        mode=$OPTARG
        [[ ! $mode =~ CUDA|HIP|OpenCL|OpenMP|Serial ]] && {
            echo "Incorrect run mode provided"
            exit 1
        }
        ;;
    e)
        element=$OPTARG
        [[ ! $element =~ Tri|Tet|Quad|Hex ]] && {
            echo "Incorrect element type provided"
            exit 1
        }
        ;;
    n)
        ndofs=$OPTARG
        ;;
    a)
        affine=true
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

if [ "$affine" = true ] ; then
    exe+=" --affine "
fi

echo "Running BK6..."

for p in {1..14}
do
    #compute mesh size
    if [ "$element" == "Hex" ] || [ "$element" == "Tet" ]; then
        N=$(echo $ndofs $p | awk '{ printf "%3.0f", ($1/(3*$2*$2*$2))^(1/3)+0.499 }')
        $mpi $exe -m $mode -e $element -nx $N -ny $N -nz $N -p $p
    elif [ "$element" == "Quad" ] || [ "$element" == "Tri" ]; then
        N=$(echo $ndofs $p | awk '{ printf "%3.0f", ($1/(2*$2*$2))^(1/2)+0.499 }')
        $mpi $exe -m $mode -e $element -nx $N -ny $N -p $p
    fi
done

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
