#!/bin/bash -e

if [[ ! $OSTYPE =~ "linux" ]]; then
    echo "Only supported on Linux"
    exit 1
fi

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd)"
cd $SCRIPTPATH

echo "Baseline from task 0.3:"
echo ""
echo "Wall-time for laplacian1Kernel with 1 iteration"
time (./main before.bmp after.bmp -i 1)

echo ""
echo "Wall-time for laplacian1Kernel with 1024 iteration"
time (./main before.bmp after.bmp -i 1024)