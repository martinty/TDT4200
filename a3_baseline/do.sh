#!/bin/bash -e

if [[ ! $OSTYPE =~ "linux" ]]; then
    echo "Only supported on Linux"
    exit 1
fi

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd)"
cd $SCRIPTPATH

echo "Baseline from task 0.3:"
echo ""
echo "Wall-time with laplacian1Kernel and 1 iteration"
time (./main before.bmp after.bmp -i 1)

echo ""
echo "Wall-time with laplacian1Kernel and 1024 iterations"
time (./main before.bmp after.bmp -i 1024)