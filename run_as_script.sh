#!/usr/bin/env bash

# ensure one arg and that it is *.ipynb

if [ "$#" -ne 1 ]; then
    echo "Usage: run_as_script.sh <notebook.ipynb>"
    exit 1
fi

# check if the file is a Jupyter notebook

if [[ $1 != *.ipynb ]]; then
    echo "Usage: run_as_script.sh <notebook.ipynb>"
    exit 1
fi


# NEWNAME=$(echo $1 | sed 's/\.ipynb//')
## Replace .ipynb with .py
NEWNAME=$(echo $1 | sed 's/\.ipynb/\.py/')
echo $NEWNAME
./notebook_to_script.sh $1  && python $NEWNAME