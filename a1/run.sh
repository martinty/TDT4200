#!/bin/bash
clear

echo "------------- Running cmake -------------"
mkdir -p build
cd build
cmake ..
make -j4

echo "------------- Running program -----------"
cd ../program
./bitmap