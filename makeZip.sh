#!/bin/bash -e

ZIPLIST="zipList.txt"

if [[ $# < 1 ]]; then
    exit
fi

FOLDER=$1

find $FOLDER/src -type f > $FOLDER/$ZIPLIST
find $FOLDER/inc -type f >> $FOLDER/$ZIPLIST
find $FOLDER -name "run.sh" >> $FOLDER/$ZIPLIST
find $FOLDER -name "CMakeLists.txt" >> $FOLDER/$ZIPLIST

for ARG in ${@:2} 
do
    find $FOLDER/program/ -name "$ARG" >> $FOLDER/$ZIPLIST
done

zip -v a1.zip -@ < $FOLDER/$ZIPLIST