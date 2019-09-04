#!/bin/bash -e

if [[ ! $OSTYPE =~ "linux" ]]; then
    echo "Only supported on Linux"
    exit 1
fi

N=${2:-1}
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd)"
cd $SCRIPTPATH

function make_f {
    echo "------------- Running make --------------"
    make
}
function run_f {
    cd program 
    echo "------------- Running program -----------"
    echo "Using $N processes"
    mpirun -np $N *.out
}
function clean_f {
    echo "------------- Running clean -------------"
    make clean
}

if [[ $1 == "build" ]]; then
    build_f
elif [[ $1 == "make" ]]; then
    make_f
elif [[ $1 == "run" ]]; then
    run_f
elif [[ $1 == "all" ]]; then
    make_f
    run_f
elif [[ $1 == "clean" ]]; then
    clean_f
else
    echo "Commands: "
    echo "      make"
    echo "      run <n>"
    echo "      all <n>"
    echo "      clean"
fi