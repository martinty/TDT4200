#!/bin/bash
clear

echo "------------- Running cmake -------------"
mkdir -p build
mkdir -p program
cd build
cmake ..
make -j4

echo "------------- Running program -----------"
cd ../program
./bitmap