#!/bin/bash

function HELP {
  echo "Usage: ./runBK.sh -m MODE -e ELEMENT -n NDOFS"
  exit 1
}

#defaults
element=Hex
ndofs=4000000

#parse options
while getopts :m:e:n:h FLAG; do
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

cd BK1; ./runBK1.sh -m $mode -e $element -n $ndofs; cd ..
cd BK2; ./runBK2.sh -m $mode -e $element -n $ndofs; cd ..
cd BK3; ./runBK3.sh -m $mode -e $element -n $ndofs; cd ..
cd BK4; ./runBK4.sh -m $mode -e $element -n $ndofs; cd ..
cd BK5; ./runBK5.sh -m $mode -e $element -n $ndofs; cd ..
cd BK6; ./runBK6.sh -m $mode -e $element -n $ndofs; cd ..


#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
