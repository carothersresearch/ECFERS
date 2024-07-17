# ECFERS
 Exploring Cell-Free Enzyme Reaction Systems 

 hello!

## FRENDA: Fast Retrieval of ENzyme DAta
Given a list of EC numbers, parses through BRENDA to generate a spreadsheet of relevant reaction data ready to feed into **ODBM**. Uses KEGG ID's to standardize chemical compounds, enzyme names, and reaction IDs. If a kinetic parameter is unavailable, it uses the predicted data found in *other folders*.

### File Formatting
If your proteomics information comes in the form of a RefSeq accession number, your file should be formatted as indicated in `proteome_exe.csv`.

Before use with `parser.sh`, your proteomics data should be formatted as follows. This will be automatically formatted if `refseq2ID.sh` is used:

| Accession Number | EC | Species | Label | Enzyme | Reaction ID | Substrates | Products  | Km | Kcat |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| WP_000789986.1 | 5.3.1.9 |  |  |  |  |  |  |  |  |

If species is left blank, it will be assumed as *Escherichia coli*. If an enzyme is exogenous, the species should be specified as is on BRENDA. The accession number may be left blank.

### Setup

Due to licensing issues, BRENDA is not able to be downloaded directly by this program, so you are asked to download it as a text file [here](https://www.brenda-enzymes.org/download.php). Convert it to a txt file and rename brenda_download:

```sh
tar -xf brenda_download.txt.tar.gz
```

Add the text file in the same directory as the project.

### Usage

Add your proteomics data in the same directory as the project. If it is in the format of RefSeq Accession Numbers (exe. WP_032359739.1), run the following at the command line to convert into EC numbers:

```sh
bash refseq2ID.sh
```

Once the output.csv file is formatted correctly with EC numbers and appropriate headers, run the following at the command line to detail the reaction information for each enzyme. The documents `Reaction.csv` 
and `SpeciesBaseMechanism.csv` now contain all relevant information formatted appropriately for input for ODBM.

```sh
bash parser.sh
```
