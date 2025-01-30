import argparse
import tellurium as te
from equilibrator_api import ComponentContribution

cc = ComponentContribution()

def parse_input_file(file_path):
    """Reads a text file and extracts unique species."""
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    species_set = set()
    for line in content:
        if "->" in line:
            parts = line.split(":")
            if len(parts) > 1:
                reaction_part = parts[1].strip().replace(";", "")
                left, right = reaction_part.split("->")
                left_words = set(left.split())
                right_words = set(right.split())
                common_words = left_words.intersection(right_words)
                species_set.update(word for word in common_words if not (word.isdigit() or word in ["+"]))
    return list(species_set)

def pull_tell_specs(file_path):
    """Pulls all floating species IDs from Antimony file"""
    with open(file_path, 'r') as file:
        antimony = file.read()
    r  = te.loada(antimony)
    allSpecies = r.getFloatingSpeciesIds()
    return allSpecies

def annotate_species(species_list, database):
    """Annotates species using the specified database."""
    annotations = []
    for spc in species_list:
        relid = None
        identity = None

        for i in cc.search_compound(spc).identifiers:
            if relid is None and i.registry.namespace == 'metacyc.compound':
                relid = i.accession

            if identity is None and i.registry.namespace == database:
                identity = i.accession

            if relid is not None and identity is not None:
                break

        annotations.append((spc, relid, identity))
    return annotations

def write_annotations(file_path, annotations, database):
    """Appends annotations to the input file."""
    with open(file_path, 'a') as file:
        file.write('\n')  # Add a newline before appending
        for spc, relid, identity in annotations:
            file.write(f'{spc} is "{relid}";\n')
            file.write(f'{spc} identity "http://identifiers.org/{database}/{identity}";\n')

def main():
    parser = argparse.ArgumentParser(description="Annotate species in a text file.")
    parser.add_argument('file_path', help="Path to the input text file.")
    parser.add_argument(
        '--database', 
        choices=['kegg', 'bigg.metabolite', 'chebi', 'hmdb', 'metacyc.compound'], 
        default='kegg', 
        help="Annotation database to use (default: kegg)."
    )
    args = parser.parse_args()

    # Process input file
    enzyme_list = parse_input_file(args.file_path)

    # Process Antimony file
    all_species_list = pull_tell_specs(args.file_path)

    # Remove enzymes from species list
    species_list = list(set(all_species_list)-set(enzyme_list))
    
    # Annotate species
    annotations = annotate_species(species_list, args.database)

    # Write annotations
    write_annotations(args.file_path, annotations, args.database)

if __name__ == "__main__":
    main()
