#!/bin/bash

function HELP {
  echo "Usage: ./run.sh -m MODE"
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

cd BK; ./runBK.sh -m $mode; cd ..
cd BP; ./runBP.sh -m $mode; cd ..


#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
