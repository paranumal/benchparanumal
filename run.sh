#!/bin/bash

mpi="mpirun -np 1 "
exe=./BP

function HELP {
    echo "Usage: ./run.sh -m MODE -e ELEMENT -n NDOFS -P problem -p degree [-k] [-a] [-t] [-v]"
    exit 1
}

function RUN {
    if [ "$element" == "Hex" ] || [ "$element" == "Tet" ]; then
        N=$(echo $ndofs $p $Nfields | awk '{ printf "%3.0f", ($1/($3*$2*$2*$2))^(1/3)+0.499 }')
        $mpi $exe -m $mode -P $problem -e $element -nx $N -ny $N -nz $N -p $p
    elif [ "$element" == "Quad" ] || [ "$element" == "Tri" ]; then
        N=$(echo $ndofs $p $Nfields | awk '{ printf "%3.0f", ($1/($3*$2*$2))^(1/2)+0.499 }')
        $mpi $exe -m $mode -P $problem -e $element -nx $N -ny $N -p $p
    fi
}

function CHECKDEGREE {
    if [ "$affine" = true ]; then
        if [ "$element" = "Tri" ]; then
            if   [ "$problem" = 1 ]; then pmax=15
            elif [ "$problem" = 2 ]; then pmax=15
            elif [ "$problem" = 3 ]; then pmax=15
            elif [ "$problem" = 4 ]; then pmax=15
            elif [ "$problem" = 5 ]; then pmax=15
            elif [ "$problem" = 6 ]; then pmax=15
            fi
        elif [ "$element" = "Quad" ]; then
            if   [ "$problem" = 1 ]; then pmax=15
            elif [ "$problem" = 2 ]; then pmax=15
            elif [ "$problem" = 3 ]; then pmax=15
            elif [ "$problem" = 4 ]; then pmax=15
            elif [ "$problem" = 5 ]; then pmax=15
            elif [ "$problem" = 6 ]; then pmax=15
            fi
        elif [ "$element" = "Tet" ]; then
            if   [ "$problem" = 1 ]; then pmax=9
            elif [ "$problem" = 2 ]; then pmax=9
            elif [ "$problem" = 3 ]; then pmax=9
            elif [ "$problem" = 4 ]; then pmax=9
            elif [ "$problem" = 5 ]; then pmax=9
            elif [ "$problem" = 6 ]; then pmax=9
            fi
        elif [ "$element" = "Hex" ]; then
            if   [ "$problem" = 1 ]; then pmax=15
            elif [ "$problem" = 2 ]; then pmax=11
            elif [ "$problem" = 3 ]; then pmax=15
            elif [ "$problem" = 4 ]; then pmax=10
            elif [ "$problem" = 5 ]; then pmax=15
            elif [ "$problem" = 6 ]; then pmax=15
            fi
        fi
    else
        if [ "$element" = "Tri" ]; then
            if   [ "$problem" = 1 ]; then pmax=15
            elif [ "$problem" = 2 ]; then pmax=15
            elif [ "$problem" = 3 ]; then pmax=15
            elif [ "$problem" = 4 ]; then pmax=15
            elif [ "$problem" = 5 ]; then pmax=15
            elif [ "$problem" = 6 ]; then pmax=15
            fi
        elif [ "$element" = "Quad" ]; then
            if   [ "$problem" = 1 ]; then pmax=15
            elif [ "$problem" = 2 ]; then pmax=15
            elif [ "$problem" = 3 ]; then pmax=15
            elif [ "$problem" = 4 ]; then pmax=15
            elif [ "$problem" = 5 ]; then pmax=15
            elif [ "$problem" = 6 ]; then pmax=15
            fi
        elif [ "$element" = "Tet" ]; then
            if   [ "$problem" = 1 ]; then pmax=9
            elif [ "$problem" = 2 ]; then pmax=9
            elif [ "$problem" = 3 ]; then pmax=9
            elif [ "$problem" = 4 ]; then pmax=8
            elif [ "$problem" = 5 ]; then pmax=9
            elif [ "$problem" = 6 ]; then pmax=9
            fi
        elif [ "$element" = "Hex" ]; then
            if   [ "$problem" = 1 ]; then pmax=15
            elif [ "$problem" = 2 ]; then pmax=10
            elif [ "$problem" = 3 ]; then pmax=15
            elif [ "$problem" = 4 ]; then pmax=9
            elif [ "$problem" = 5 ]; then pmax=15
            elif [ "$problem" = 6 ]; then pmax=15
            fi
        fi
    fi
}


#defaults
element=Hex
ndofs=4000000
affine=false
kernel=false
tune=false
verbose=false
problem=-1
p=-1

#parse options
while getopts :m:e:n:P:p:aktvh FLAG; do
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
    P)
        problem=$OPTARG
        ;;
    p)
        p=$OPTARG
        ;;
    a)
        affine=true
        ;;
    k)
        kernel=true
        ;;
    t)
        tune=true
        ;;
    v)
        verbose=true
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
if [ "$kernel" = true ] ; then
    exe+=" -k "
fi
if [ "$tune" = true ] ; then
    exe+=" -t "
fi
if [ "$verbose" = true ] ; then
    exe+=" -v "
fi

if [ "$problem" = 2 ] || [ "$problem" = 4 ] || [ "$problem" = 6 ]; then
    if [ "$element" = "Hex" ] || [ "$element" = "Tet" ]; then
        Nfields=3
    else
        Nfields=2
    fi
else
    Nfields=1
fi

if [ "$problem" = -1 ] ; then
    for problem in $(seq 1 6)
    do
        if [ "$p" = -1 ] ; then
            pmin=1
            pmax=15

            CHECKDEGREE

            for p in $(seq $pmin $pmax)
            do
                RUN
            done
            p=-1
        else
            RUN
        fi
    done
else
    if [ "$p" = -1 ] ; then
        pmin=1
        pmax=15

        CHECKDEGREE

        for p in $(seq $pmin $pmax)
        do
            RUN
        done
    else
        RUN
    fi
fi

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
