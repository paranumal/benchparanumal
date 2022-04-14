#!/bin/bash

function HELP {
  echo "Usage: ./runBP.sh -m MODE"
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

cd BP1; ./runBP1.sh -m $mode; cd ..
cd BP2; ./runBP2.sh -m $mode; cd ..
cd BP3; ./runBP3.sh -m $mode; cd ..
cd BP4; ./runBP4.sh -m $mode; cd ..
cd BP5; ./runBP5.sh -m $mode; cd ..
cd BP6; ./runBP6.sh -m $mode; cd ..

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
