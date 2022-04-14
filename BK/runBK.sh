#!/bin/bash

function HELP {
  echo "Usage: ./runBK.sh -m MODE"
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

cd BK1; ./runBK1.sh -m $mode; cd ..
cd BK2; ./runBK2.sh -m $mode; cd ..
cd BK3; ./runBK3.sh -m $mode; cd ..
cd BK4; ./runBK4.sh -m $mode; cd ..
cd BK5; ./runBK5.sh -m $mode; cd ..
cd BK6; ./runBK6.sh -m $mode; cd ..

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
