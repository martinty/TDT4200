#!/bin/bash
clear

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd)"
cd $SCRIPTPATH

echo "------------- Running cmake -------------"
mkdir -p build
mkdir -p program
cd build
cmake ..
make -j4

echo "------------- Running program -----------"
cd ../program
./bitmap