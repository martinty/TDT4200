#!/bin/bash -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd)"
cd $SCRIPTPATH

if [[ $1 == "new" ]]; then
    git clean -dfx
elif [[ $1 == "clean" ]]; then
    cd build
    make clean
    exit 0
fi

mkdir -p build
mkdir -p program
cd build

if [[ $1 != "run" ]]; then
    echo "------------- Running cmake -------------"
    cmake ..
fi

if [[ $OSTYPE =~ "linux" ]]; then
    echo "------------- Running make --------------"
    make -j4
    echo "------------- Running program -----------"
    cd ../program
    ./bitmap
else
    echo "$OSTYPE is not supported for make and run!"
fi