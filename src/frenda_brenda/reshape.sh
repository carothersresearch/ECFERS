#!/bin/bash

# Check if three arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <filename> <file2> <file3>"
    exit 1
fi

# Extract filename and other filenames from command line arguments
sbml_base="$1"
reaction_base="$2"
sbm_base="$3"

# Check if other files exist
if [ ! -f "$reaction_base" ] || [ ! -f "$sbm_base" ]; then
    echo "Error: One or more files do not exist"
    exit 1
fi

# Call the Python script with the provided filenames as arguments
python reshape.py "$sbml_base" "$reaction_base" "$sbm_base"
