#!/bin/bash

ZIPLIST="zipList.txt"

if [[ $# < 1 ]]; then
    exit
fi

FOLDER=$1
find $FOLDER/src -type f > $FOLDER/$ZIPLIST
find $FOLDER/inc -type f >> $FOLDER/$ZIPLIST
find $FOLDER -name "run.sh" >> $FOLDER/$ZIPLIST
find $FOLDER -name "do.sh" >> $FOLDER/$ZIPLIST
find $FOLDER -name "CMakeLists.txt" >> $FOLDER/$ZIPLIST
find $FOLDER -maxdepth 1 -name "Makefile" >> $FOLDER/$ZIPLIST

for ARG in ${@:2} 
do
    find $FOLDER/program/ -name "$ARG" >> $FOLDER/$ZIPLIST
done

ARCHIVE="$FOLDER.zip"
rm -f $ARCHIVE
zip -v $ARCHIVE -@ < $FOLDER/$ZIPLIST
