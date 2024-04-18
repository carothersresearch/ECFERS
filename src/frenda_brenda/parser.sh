#!/bin/bash

# Check if a filename argument is provided
if [ -z "$1" ]; then
    python parser.py
    exit 1
fi

chosenfile=$1

# Check if file exists
if [ ! -f "$1" ]; then
    echo "$1 does not exist. Exiting..."
    exit 1
fi

# Call the Python script with the chosen file as an argument
python parserAppend.py "$chosenfile"
