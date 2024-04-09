#!/bin/bash

# Check if a filename argument is provided
if [ -z "$1" ]; then
    python parser.py
    exit 1
fi

# Define function to create CSV files with specified headers
create_csv_files() {
    echo "Accession Number,EC,Species,Label,Enzyme,Reaction ID,Mechanism,Substrates,Products,Km,Kcat,Inhibitors,KI" > Files/append_Reaction.csv
    echo "Label,EC,Type,StartingConc,Conc,Mechanisms,Parameters" > Files/append_SpeciesBaseMechanism.csv
    echo "Accession Number,Conc,EC" > Files/Inaccessible_IDs.csv
}

# Check if file exists
if [ ! -f "$1" ]; then
    echo "$1 does not exist. Exiting..."
    exit 1
fi

# If file exists, create the CSV files and populate them accordingly
create_csv_files

# Read each line of the input CSV file, skipping the first line (header)
{
    read header
    while IFS=, read -r ec species concentration; do
        # Populate append_Reaction.csv
        echo ",$ec,$species,,,,,,,," >> Files/append_Reaction.csv

        # Populate append_SpeciesBaseMechanism.csv
        echo ",$ec,,,$concentration,,,,," >> Files/append_SpeciesBaseMechanism.csv

        # Populate Inaccessible_IDs.csv
        echo ",$concentration,$ec" >> Files/Inaccessible_IDs.csv
    done
} < "$1"

python parserAppend.py
