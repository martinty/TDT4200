#!/bin/bash -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd)"
cd $SCRIPTPATH

if [[ $1 == "new" ]]; then
    git clean -dfx program/
    git clean -dfx build/
elif [[ $1 == "clean" ]]; then
    cd build
    make clean
    exit 0
fi

mkdir -p build
mkdir -p program
cd build

if [[ $1 != "run" ]] && [[ $1 != "gdb" ]]; then
    echo "------------- Running cmake -------------"
    cmake .. 
fi

if [[ $OSTYPE =~ "linux" ]]; then
    echo "------------- Running make --------------"
    make -j4
    cd ../program
    if [[ $1 == "gdb" ]]; then
        echo "------------- Running gdb ---------------"
        echo "------------- Onlye one process ---------"
        gdb ./bitmap
    elif [[ $1 == "run" ]]; then
        echo "------------- Running program -----------"
        N=${2:-1}
        mpiexec -n $N ./bitmap
    fi
else
    echo "$OSTYPE is not supported for make and run!"
fi