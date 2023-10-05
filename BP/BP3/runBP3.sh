#!/bin/bash

mpi="mpirun -np 1 "
exe="./BP3"

function HELP {
  echo "Usage: ./runBP3.sh -m MODE -e ELEMENT -n NDOFS"
  exit 1
}

#defaults
element=Hex
ndofs=4000000
affine=false
plat=0
devi=0

#parse options
while getopts :m:e:n:p:d:ah FLAG; do
  case $FLAG in
    m)
        mode=$OPTARG
        [[ ! $mode =~ CUDA|HIP|OpenCL|OpenMP|Serial|DPCPP ]] && {
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
    p)
        plat=$OPTARG;
	echo "platform=" $plat;;
    d)
        devi=$OPTARG;
	echo "device=" $devi;;

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

echo "Running BP3..."

for p in {1..14}
do
    #compute mesh size
    if [ "$element" == "Hex" ] || [ "$element" == "Tet" ]; then
        N=$(echo $ndofs $p | awk '{ printf "%3.0f", ($1/($2*$2*$2))^(1/3)+0.499 }')

	echo $mpi $exe " -m " $mode " -e " $element " -nx " $N " -ny " $N " -nz " $N " -p " $p " -pl " $plat " -d " $devi
	
        $mpi $exe -m $mode -e $element -nx $N -ny $N -nz $N -p $p -pl $plat -d $devi
    elif [ "$element" == "Quad" ] || [ "$element" == "Tri" ]; then
        N=$(echo $ndofs $p | awk '{ printf "%3.0f", ($1/($2*$2))^(1/2)+0.499 }')
        $mpi $exe -m $mode -e $element -nx $N -ny $N -p $p -pl $plat -d $devi
    fi
done


#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
