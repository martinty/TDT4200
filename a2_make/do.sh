#!/bin/bash -e


SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd)"
cd $SCRIPTPATH


if [[ $1 == "c" ]]; then
    make clean
    exit 0
fi


make
N=${1:-0}
if [[ $N > 0 ]]; then
    make run n=$N
fi