#!/usr/bin/env bash

# ensure one arg and that it is *.ipynb
if [ "$#" -ne 1 ]; then
    echo "Usage: notebook_to_script.sh <notebook.ipynb>"
    exit 1
fi

# check if the file is a Jupyter notebook
if [[ $1 != *.ipynb ]]; then
    echo "Usage: notebook_to_script.sh <notebook.ipynb>"
    exit 1
fi

# This script converts a Jupyter notebook to a Python script
jupyter nbconvert --to script $1
