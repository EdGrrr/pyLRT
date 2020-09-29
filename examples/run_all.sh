#! /bin/bash

for i in *.py;
do
    echo $i;
    python $i > /dev/null;
done
