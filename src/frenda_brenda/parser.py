import gzip
import json
import os
import sys
import math

import numpy as np
import pandas as pd

from brendapyrser import BRENDA
from equilibrator_api import ComponentContribution
from itertools import filterfalse

# MAKE SURE THESE FILE PATHS ARE UPDATED AND THAT THE FILES EXIST IN THE REPO
path = os.getcwd()

with gzip.open(path+"../thermo_calculations/kegg_enzymes.json.gz", "r") as f:
        ECs = {e['EC']:e['reaction_ids'] for e in json.load(f)}

with gzip.open(path+"../thermo_calculations/kegg_reactions.json.gz", "r") as f:
        RXNs = {r['RID']:r['reaction'] for r in json.load(f)}

reaction = pd.read_csv(path+'/Files/Reaction.csv')
sbp = pd.read_csv(path+'/Files/SpeciesBaseMechanisms.csv')
kcats = pd.read_csv(path+'../kinetic_estimator/full_report_kcats.csv') #GET THIS FROM WHERE?
kms = pd.read_csv(path+'../kinetic_estimator/full_report_kms.csv') #GET THIS FROM WHERE?
dataFile = path+'/Files/brenda_download.txt'

brenda = BRENDA(dataFile)
cc = ComponentContribution()
compound_dict = {}

class ECIndexError(Exception):
    pass

def get_enzyme_name(reaction):
    return reaction.name

def get_substrates_and_products(EC):
    try:
        rxn_IDs = ECs[EC]
    except KeyError:
        raise ECIndexError("This EC is not present in the KEGG database")

    rxn_comp = []

    for rxn in rxn_IDs:
        try:
            substrate = [species[1] for species in RXNs[rxn] if species[0] < 0]
            products = [species[1] for species in RXNs[rxn] if species[0] >= 0]
        except KeyError:
            print('Could not index ',rxn)
            continue
        rxn_comp.append([rxn, substrate, products])

    df = pd.DataFrame(rxn_comp, columns=['Reaction ID', 'Substrates', 'Products'])

    # Remove the brackets from the DataFrame
    df = df.applymap(lambda x: '; '.join(x) if isinstance(x, list) else x)

    return df

def get_parameters(rxn, species):

    parameters_list = []

    # gets list of all KM and Kcat values for the organism
    kmvals = rxn.KMvalues.filter_by_organism(species)
    kcatvals = rxn.Kcatvalues.filter_by_organism(species)

    # pulls the substrates from both lists and combines them with no duplicates
    substrate_list_km = []
    for i in kmvals:
        substrate_list_km.append(i)

    substrate_list_kcat = []
    for i in kcatvals:
        substrate_list_kcat.append(i)

    substrate_list = list(set(substrate_list_km + substrate_list_kcat))

    for compound in substrate_list:
        KMfiltered = kmvals.filter_by_compound(compound).get_values()
        KMMfiltered = rxn.KKMvalues.filter_by_organism(species).filter_by_compound(compound).get_values()
        Kcatfiltered = kcatvals.filter_by_compound(compound).get_values()
        KMfiltered = list(filterfalse(KMMfiltered.__contains__, KMfiltered))

        KMvals_WT = []

        if len(KMfiltered) > 1:
            kval = kmvals.filter_by_compound(compound)

            metalist = []
            for k in kval.keys():
                for v in kval[k]:
                    metalist.append(v['meta'])

            for idx, meta in enumerate(metalist):
                if "wild" in meta:
                    KMvals_WT.append(kval[k][idx]['value'])

        Kcatvals_WT = []
        if len(Kcatfiltered) > 1:
            kval = kcatvals.filter_by_compound(compound)

            metalist = []
            for k in kval.keys():
                for v in kval[k]:
                    metalist.append(v['meta'])

            for idx, meta in enumerate(metalist):
                if "wild" in meta:
                    Kcatvals_WT.append(kval[k][idx]['value'])

        if len(KMvals_WT) != 0:
            KMfiltered = KMvals_WT

        if len(Kcatvals_WT) != 0:
            Kcatfiltered = Kcatvals_WT

        KM = sum(KMfiltered)/(len(KMfiltered) or 1)
        KMM = sum(KMMfiltered)/(len(KMMfiltered) or 1)
        Kcat = sum(Kcatfiltered)/(len(Kcatfiltered) or 1)

        parameters_list.append([compound, KM, Kcat])

    return parameters_list

def get_parameters_KEGG(parameters):
    kegg_params = []

    for compound in parameters:
        try:
            keggid = compound_dict[compound[0]]
        except KeyError:
            try:
                for i in cc.search_compound(compound[0]).identifiers:
                    if i.registry.namespace == 'kegg':
                        keggid = i.accession
                        compound_dict[compound[0]] = keggid
            except ValueError:
                print('There were no search results found for compound ', compound[0])
                continue
        try:
            keggid
        except NameError:
            keggid = 'NA'

        kegg_params.append([f'Km_{keggid}: {compound[1]}; Kcat_{keggid}: {compound[2]}'])

    return kegg_params

def grab_KM_prediction(EC, substrate):
    condition = (kms['enzyme'] == EC) & (kms['substrates'] == substrate)
    matching_indices = kms.index[condition]

    i = matching_indices[0]
    km = kms.loc[i, 'Km']

    return km

def grab_Kcat_prediction(EC, substrate):
    condition = (kcats['enzyme'] == EC) & (kcats['substrates'] == substrate)
    matching_indices = kcats.index[condition]

    i = matching_indices[0]
    kcat = kcats.loc[i, 'kcats']

    return kcat

# Function to extract Km and Kcat values
def extract_values(EC, entries, km_dict, kcat_dict):
    km_values = []
    kcat_values = []

    for entry in entries:
        if entry == '':
            continue
        km_key = f'Km_{entry}'
        kcat_key = f'Kcat_{entry}'
        km_value = f'{km_key}: {km_dict.get(km_key, grab_KM_prediction(EC, entry))}'
        kcat_value = f'{kcat_key}: {kcat_dict.get(kcat_key, grab_Kcat_prediction(EC, entry))}'
        km_values.append(km_value)
        kcat_values.append(kcat_value)
    return km_values, kcat_values

# Function to make a dataframe with all relevant data for a single EC number
def assemble(ID, species):
    try:
        rxn = brenda.reactions.get_by_id(ID)
    except ValueError:
        raise ECIndexError('This EC was not found in the BRENDA database')

    name = get_enzyme_name(rxn)

    try:
        sub_prod = get_substrates_and_products(ID)
    except ECIndexError:
        raise ECIndexError("This EC is not present in the KEGG database")


    sub_prod.insert(0, 'Enzyme', name)

    kegg_params = get_parameters_KEGG(get_parameters(rxn, species))

    km_dict = {}
    kcat_dict = {}
    for entry in kegg_params:
        entry_values = entry[0].split('; ')
        for value in entry_values:
            key, val = value.split(': ')
            if key.startswith('Km_'):
                km_dict[key] = val
            elif key.startswith('Kcat_'):
                kcat_dict[key] = val

    kcat_dict = {x:y for x,y in kcat_dict.items() if not math.isclose(float(y), 0)}
    km_dict = {x:y for x,y in km_dict.items() if not math.isclose(float(y), 0)}

    for kcat in kcat_dict:
        if sub_prod['Substrates'].str.contains(kcat[5:]).any():
            KcatF = kcat_dict[kcat]
        elif sub_prod['Products'].str.contains(kcat[5:]).any():
            KcatR = kcat_dict[kcat]
        else:
            KcatF = 0
            KcatR = 0

    sub_prod[['Km', 'Kcat']] = sub_prod.apply(lambda row: pd.Series(extract_values(ID, row['Products'].split('; ') + row['Substrates'].split('; '), km_dict, kcat_dict)), axis=1)

    sub_prod['Km'] = sub_prod['Km'].apply('; '.join)
    sub_prod['Kcat'] = sub_prod['Kcat'].apply('; '.join)

    return sub_prod

def calculate_kcat_F_and_kcat_R(df):
    # Initialize empty lists to store Kcat_F and Kcat_R values
    kcat_F_values = []
    kcat_R_values = []

    for index, row in df.iterrows():
        substrates = row['Substrates'].split('; ')
        products = row['Products'].split('; ')
        kcat_values = row['Kcat'].split('; ')

        # Initialize variables to calculate the sum and count for Kcat_F and Kcat_R
        kcat_F_sum = 0
        kcat_F_count = 0
        kcat_R_sum = 0
        kcat_R_count = 0

        for kcat_value in kcat_values:
            kcat_suffix = kcat_value.split(': ')[0][5:]  # Extract the suffix (e.g., C00417)
            kcat = float(kcat_value.split(': ')[1])

            if kcat_suffix in substrates and not np.isnan(kcat):
                kcat_F_sum += kcat
                kcat_F_count += 1
            elif kcat_suffix in products and not np.isnan(kcat):
                kcat_R_sum += kcat
                kcat_R_count += 1

        # Calculate Kcat_F and Kcat_R averages or set to NaN if no matching values
        kcat_F_average = kcat_F_sum / kcat_F_count if kcat_F_count > 0 else np.nan
        kcat_R_average = kcat_R_sum / kcat_R_count if kcat_R_count > 0 else np.nan

        # Append the calculated values to the respective lists
        kcat_F_values.append(kcat_F_average)
        kcat_R_values.append(kcat_R_average)

    # Create new columns for Kcat_F and Kcat_R in the DataFrame
    df['Kcat_F'] = kcat_F_values
    df['Kcat_R'] = kcat_R_values

    # Replace the 'Kcat' column with 'Kcat_F' and 'Kcat_R' values
    df['Kcat'] = df.apply(lambda row: f'Kcat_F: {row["Kcat_F"]}; Kcat_R: {row["Kcat_R"]}', axis=1)

    # Remove the individual 'Kcat_F' and 'Kcat_R' columns
    df.drop(['Kcat_F', 'Kcat_R'], axis=1, inplace=True)

    return df

# Function that iterates through Reaction.csv and pulls the relevant data for each EC using prior functions
def iterate(reaction_df, sbp_df):
    expanded_dfs = []
    for index, row in reaction_df.iterrows():
        ec = row['EC']
        species = row['Species']

        print('Processing EC ',ec)

        if np.isnan(species):
            species = 'Escherichia coli'

        try:
            extracted_df = assemble(ec, species)
        except ECIndexError:
            print('Could not index EC ', ec)
            continue

        sbp_df.iloc[index,0] = get_enzyme_name(brenda.reactions.get_by_id(ec))

        # Add the 'Accession Number', 'EC', 'Species', and 'Label' from the original row to the extracted data
        extracted_df['Accession Number'] = row['Accession Number']
        extracted_df['EC'] = row['EC']
        extracted_df['Species'] = row['Species']
        extracted_df['Label'] = row['Label']

        expanded_dfs.append(extracted_df)

    # Concatenate the extracted dataframes and reorganize the columns
    expanded_df = pd.concat(expanded_dfs, ignore_index=True)

    # Reorder the columns to match the original dataframe
    column_order = reaction_df.columns
    expanded_df = expanded_df[column_order]

    expanded_df = calculate_kcat_F_and_kcat_R(expanded_df)

    for index, row in expanded_df.iterrows():
        expanded_df.iloc[index, 3] = f'R{index+1}'

    return expanded_df, sbp_df

# THIS NEEDS CHANGED TO BE UPDATED FOR CURRENT PRACTICES
def main():
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    dataFile = sys.argv[3]
    # filtoption = sys.argv[4]

    print('this is running')

    # This reads in Reaction.csv as df and SpeciesBaseMechanisms.csv as df2
    df = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    # df, df2 = rmBlanks(df, df2)

    brenda = BRENDA(dataFile)

    parsedRXNs, parsedSBM = iterate(df, df2)

    # Edits the parsed Reaction.csv in place
    parsedRXNs.to_csv(filename1, index=False)
    parsedSBM.to_csv(filename2, index=False)

if __name__ == '__main__':
    main()
