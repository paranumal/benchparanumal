#!/bin/bash

function HELP {
  echo "Usage: ./runBK.sh -m MODE -e ELEMENT -n NDOFS"
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

args="-m $mode -e $element -n $ndofs"

if [ "$affine" = true ] ; then
    args+=" -a "
fi

cd BK1; ./runBK1.sh $args; cd ..
cd BK2; ./runBK2.sh $args; cd ..
cd BK3; ./runBK3.sh $args; cd ..
cd BK4; ./runBK4.sh $args; cd ..
cd BK5; ./runBK5.sh $args; cd ..
cd BK6; ./runBK6.sh $args; cd ..


#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
