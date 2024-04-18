import gzip
import json
import os
import sys
import math
import re

import numpy as np
import pandas as pd

from brendapyrser import BRENDA
from equilibrator_api import ComponentContribution
from itertools import filterfalse

path = os.getcwd()

chosenfile = pd.read_csv(sys.argv[1])

with gzip.open(path+"/../thermo_calculations/kegg_enzymes.json.gz", "r") as f:
        ECs = {e['EC']:e['reaction_ids'] for e in json.load(f)}

with gzip.open(path+"/../thermo_calculations/kegg_reactions.json.gz", "r") as f:
        RXNs = {r['RID']:r['reaction'] for r in json.load(f)}

reaction_og = pd.read_csv(path+'/Files/Reaction.csv')
sbm_og = pd.read_csv(path+'/Files/SpeciesBaseMechanisms.csv')
inac = pd.read_csv(path+'/Files/Inaccessible_IDs.csv')
kcats = pd.read_csv(path+'/../kinetic_estimator/full_report_kcats.csv') #GET THIS FROM WHERE?
kms = pd.read_csv(path+'/../kinetic_estimator/full_report_kms.csv') #GET THIS FROM WHERE?
kis = pd.read_csv(path+'/../kinetic_estimator/full_report_kis.csv') #GET THIS FROM WHERE?
metabconc_ref = pd.read_csv(path+'/Files/Metabolite_Concentrations.csv')
dataFile = path+'/Files/brenda_download.txt'


brenda = BRENDA(dataFile)
cc = ComponentContribution()
compound_dict = {}

class ECIndexError(Exception):
    pass


LastLabel = reaction_og["Label"].iloc[-1]

print(LastLabel)

# Create reaction DataFrame
reaction_toappend = pd.DataFrame(columns=['Accession Number', 'EC', 'Species', 'Label', 'Enzyme', 'Reaction ID',
                                  'Mechanism', 'Substrates', 'Products', 'Km', 'Kcat', 'Inhibitors', 'KI'])
# Create sbm DataFrame
sbm_toappend = pd.DataFrame(columns=['Label', 'EC', 'Type', 'StartingConc', 'Conc', 'Mechanisms', 'Parameters'])

# Function to generate Label values
def generate_label():
    global LastLabel
    LastLabel = 'R' + str(int(LastLabel[1:]) + 1)
    return LastLabel

# Iterate over each row in the DataFrame
for index, row in chosenfile.iterrows():
    if row['Type'] == 'Enzyme':
        # Add row to reaction DataFrame
        reaction_toappend = reaction_toappend.append({
            'Accession Number': '',
            'EC': row['EC'],
            'Species': row['Species'],
            'Label': generate_label(),
            'Enzyme': '',
            'Reaction ID': '',
            'Mechanism': '',
            'Substrates': '',
            'Products': '',
            'Km': '',
            'Kcat': '',
            'Inhibitors': '',
            'KI': ''
        }, ignore_index=True)

        # Add row to sbm DataFrame
        sbm_toappend = sbm_toappend.append({
            'Label': '',
            'EC': row['EC'],
            'Type': 'Enzyme',
            'StartingConc': row['Concentration'],
            'Conc': '',
            'Mechanisms': '',
            'Parameters': ''
        }, ignore_index=True)

        print(LastLabel)

    elif row['Type'] == 'Metabolite':
        # Add row to sbm DataFrame
        sbm_toappend = sbm_toappend.append({
            'Label': row['KEGG'],
            'EC': '',
            'Type': 'Metabolite',
            'StartingConc': row['Concentration'],
            'Conc': '',
            'Mechanisms': '',
            'Parameters': ''
        }, ignore_index=True)

def manualEC(sbm, reaction, inac):
    inac = inac.dropna()
    # Appending data to sbm DataFrame
    sbm_rows = []
    for index, row in inac.iterrows():
        sbm_row = pd.DataFrame({'Label': row['Accession Number'],
                                'EC': row['EC'],
                                'Type': 'Enzyme',
                                'StartingConc': row['Conc'],
                                'Conc': np.nan,
                                'Mechanisms': np.nan,
                                'Parameters': np.nan}, index=[0])
        sbm_rows.append(sbm_row)

    if sbm_rows:
        sbm = pd.concat([sbm] + sbm_rows, ignore_index=True)

    # Appending data to reaction DataFrame
    reaction_rows = []
    for index, row in inac.iterrows():
        reaction_row = pd.DataFrame({'Accession Number': row['Accession Number'],
                                     'EC': row['EC'],
                                     'Species': np.nan,
                                     'Label': np.nan,
                                     'Enzyme': np.nan,
                                     'Reaction ID': np.nan,
                                     'Mechanism': np.nan,
                                     'Substrates': np.nan,
                                     'Products': np.nan,
                                     'Km': np.nan,
                                     'Kcat': np.nan,
                                     'Inhibitors': np.nan,
                                     'KI': np.nan}, index=[0])
        reaction_rows.append(reaction_row)

    if reaction_rows:
        reaction = pd.concat([reaction] + reaction_rows, ignore_index=True)

    return sbm, reaction

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

def get_inhibitors(enzyme):
    enzyme_df = kis[kis['enzyme'] == enzyme]

    # Extract substrates and Km for the given enzyme
    substrates_list = enzyme_df['substrates'].str.split(', ').explode().tolist()
    km_list = enzyme_df['Km'].tolist()

    # Combine substrates and Km in the desired format (substrate: Km)
    substrates_km_formatted = [f"{substrate}_KI: {km}" for substrate, km in zip(substrates_list, km_list)]

    # Join the formatted strings with semicolons
    substrates_km_str = "; ".join(substrates_km_formatted)

    # Create a list with substrates separated by semicolons
    substrates_str = "; ".join(substrates_list)

    return substrates_str, substrates_km_str

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

    try:
        i = matching_indices[0]
        km = kms.loc[i, 'Km']
    except IndexError:
        print('EC ID ', EC ,' does not have corresponding kinetic parameters. Please use the kinetic estimator to find these values.')
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

    inhibitor_names, KI_values = get_inhibitors(ID)

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

    sub_prod['Inhibitors'] = inhibitor_names
    sub_prod['KI'] = KI_values

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
def iterate(reaction_df, sbm_df):
    expanded_dfs = []
    for index, row in reaction_df.iterrows():
        ec = row['EC']
        species = row['Species']

        print('Processing EC ',ec)

        if species == np.nan:
            species = 'Escherichia coli'

        try:
            extracted_df = assemble(ec, species)
        except ECIndexError:
            print('Could not index EC ', ec)
            continue

        sbm_df.iloc[index,0] = get_enzyme_name(brenda.reactions.get_by_id(ec))

        # Add the 'Accession Number', 'EC', 'Species', and 'Label' from the original row to the extracted data
        extracted_df['Accession Number'] = row['Accession Number']
        extracted_df['EC'] = row['EC']
        extracted_df['Species'] = row['Species']
        extracted_df['Label'] = row['Label']
        extracted_df['Mechanism'] = 'MRL'

        expanded_dfs.append(extracted_df)

    # Concatenate the extracted dataframes and reorganize the columns
    expanded_df = pd.concat(expanded_dfs, ignore_index=True)

    # Reorder the columns to match the original dataframe
    column_order = reaction_df.columns
    expanded_df = expanded_df[column_order]

    expanded_df = calculate_kcat_F_and_kcat_R(expanded_df)

    for index, row in expanded_df.iterrows():
        expanded_df.iloc[index, 3] = f'R{index+1}'

    return expanded_df, sbm_df

# Define a function to calculate the average excluding NaN values
def calculate_average(series):
    non_nan_values = series.dropna()
    if non_nan_values.empty:
        return np.nan
    else:
        return non_nan_values.mean()

def fillna_kcat(df):
    # Splitting 'Kcat' values into 'Kcat_F_temp' and 'Kcat_R_temp' columns
    df[['Kcat_F_temp', 'Kcat_R_temp']] = df['Kcat'].str.extract(r'Kcat_F: (.*?);.*Kcat_R: (.*?)$')

    # Convert 'Kcat_F_temp' and 'Kcat_R_temp' columns to float
    df['Kcat_F_temp'] = pd.to_numeric(df['Kcat_F_temp'], errors='coerce')
    df['Kcat_R_temp'] = pd.to_numeric(df['Kcat_R_temp'], errors='coerce')

    # Calculate the overall average of 'Kcat_F_temp' and 'Kcat_R_temp'
    average_kcat_f = calculate_average(df['Kcat_F_temp'])
    average_kcat_r = calculate_average(df['Kcat_R_temp'])

    # Replace 'nan' values in 'Kcat' with overall averages
    df['Kcat'] = df['Kcat'].str.replace('nan', f'{average_kcat_f}')

    # Drop temporary columns 'Kcat_F_temp' and 'Kcat_R_temp'
    df.drop(columns=['Kcat_F_temp', 'Kcat_R_temp'], inplace=True)

    return df

def fix_nan_Km(df):
    def replace_nan_with_avg(row):
        km_values = row['Km'].split('; ')
        non_nan_values = []
        for km_value in km_values:
            compound_id, value = km_value.split(': ')
            if value != 'nan':
                non_nan_values.append(float(value))
        avg_value = np.mean(non_nan_values)
        fixed_km_values = []
        for km_value in km_values:
            compound_id, value = km_value.split(': ')
            if value == 'nan':
                fixed_km_values.append(f"{compound_id}: {avg_value}")
            else:
                fixed_km_values.append(km_value)
        return '; '.join(fixed_km_values)

    df['Km'] = df.apply(replace_nan_with_avg, axis=1)
    return df

def fix_nan_KI(reaction):
    def replace_nan_with_avg(row, avg_value):
        if pd.isna(row['KI']):  # Check if the value is NaN
            return row['KI']

        ki_values = row['KI'].split(';')
        non_nan_values = []
        for ki_value in ki_values:
            compound_id, value = ki_value.split(': ')
            if value != 'nan':
                non_nan_values.append(float(value))

        fixed_ki_values = []
        for ki_value in ki_values:
            compound_id, value = ki_value.split(': ')
            if value == 'nan':
                fixed_ki_values.append(f"{compound_id}: {avg_value}")
            else:
                fixed_ki_values.append(ki_value)
        return ';'.join(fixed_ki_values)

    all_ki_values = []
    for ki in reaction['KI']:
        if isinstance(ki, str) and ki == '':
            continue
        if isinstance(ki, float) and np.isnan(ki):
            continue
        ki_values = ki.split(';')
        ki_values = [float(ki_value.split(': ')[1]) for ki_value in ki_values if ki_value.split(': ')[1] != 'nan']
        all_ki_values.extend(ki_values)
    overall_avg_value = np.mean(all_ki_values)

    for i, row in reaction.iterrows():
        if isinstance(row['KI'], str) and row['KI'] == '':
            continue
        if isinstance(row['KI'], float) and np.isnan(row['KI']):
            continue
        ki_values = row['KI'].split(';')
        non_nan_values = [float(ki_value.split(': ')[1]) for ki_value in ki_values if ki_value.split(': ')[1] != 'nan']
        if non_nan_values:
            avg_value = np.mean(non_nan_values)
            reaction.at[i, 'KI'] = replace_nan_with_avg(row, avg_value)
        else:
            reaction.at[i, 'KI'] = replace_nan_with_avg(row, overall_avg_value)

    return reaction

def removeNonCompounds(df):
    # Convert 'Inhibitors' column to string type to handle NaN values
    df['Inhibitors'] = df['Inhibitors'].astype(str)
    df['KI'] = df['KI'].astype(str)

    # Handle NaN values
    df['Inhibitors'] = df['Inhibitors'].apply(lambda x: np.nan if x.strip() == 'nan' else x)
    df['KI'] = df['KI'].apply(lambda x: np.nan if x.strip() == 'nan' else x)

    # Apply the helper function to filter out non-compound values
    df['Inhibitors'] = df['Inhibitors'].apply(filter_compounds)
    df['KI'] = df['KI'].apply(filter_ki_compounds)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df

# Helper function to filter out non-compound values
def filter_compounds(inhibitors):
    if pd.isna(inhibitors):
        return np.nan
    return ';'.join([i for i in inhibitors.split(';') if not i.strip().startswith(('D', 'G'))])

# Helper function to filter out non-compound values in the "KI" column
def filter_ki_compounds(ki):
    if pd.isna(ki):
        return np.nan
    return ";".join([ki_pair for ki_pair in ki.split(';') if 'D' not in ki_pair.split('_')[0] and 'G' not in ki_pair.split('_')[0]])

def removeDuplicates(df):
    # Convert 'Inhibitors' column to string type to handle NaN values
    df['Inhibitors'] = df['Inhibitors'].astype(str)
    df['KI'] = df['KI'].astype(str)

    # Handle NaN values
    df['Inhibitors'] = df['Inhibitors'].apply(lambda x: np.nan if x.strip() == 'nan' else x)
    df['KI'] = df['KI'].apply(lambda x: np.nan if x.strip() == 'nan' else x)

    # Apply the helper function to filter out non-compound values and remove duplicates
    df['Inhibitors'] = df['Inhibitors'].apply(filter_and_remove_duplicates)
    df['KI'] = df['KI'].apply(filter_and_remove_duplicates)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df

# Helper function to filter out non-compound values and remove duplicates
def filter_and_remove_duplicates(column_data):
    if pd.isna(column_data):
        return np.nan
    return ';'.join(set([item.strip() for item in column_data.split(';')]))

def drop_and_sum_duplicates(reaction, sbm):
    # Create a mask to identify rows where Type is "Metabolite"
    metabolite_mask = sbm['Type'] == 'Metabolite'

    # Handle cases where Type is "Metabolite"
    metabolite_sbm = sbm[metabolite_mask].copy()  # Create a copy to avoid SettingWithCopyWarning
    metabolite_sbm.loc[:, 'StartingConc'] = metabolite_sbm.groupby(['Label'])['StartingConc'].transform('sum')
    metabolite_sbm = metabolite_sbm.drop_duplicates(subset=['Label'])

    # Handle cases where Type is "Enzyme"
    enzyme_sbm = sbm[~metabolite_mask].copy()  # Create a copy to avoid SettingWithCopyWarning
    enzyme_sbm.loc[:, 'StartingConc'] = enzyme_sbm.groupby(['EC'])['StartingConc'].transform('sum')
    enzyme_sbm = enzyme_sbm.drop_duplicates(subset=['EC'])

    # Concatenate the results
    sbm = pd.concat([metabolite_sbm, enzyme_sbm])

    # Drop all duplicates in Reaction based on both EC AND Reaction ID
    reaction = reaction.drop_duplicates(subset=['EC', 'Reaction ID'])

    return reaction, sbm


def clean_RXN(df, sbm):
    df = fillna_kcat(df)
    df = fix_nan_Km(df)
    df = removeNonCompounds(df)
    df = removeDuplicates(df)
    df = fix_nan_KI(df)
    df, sbm = drop_and_sum_duplicates(df, sbm)

    return df, sbm

def main():
    print('this is running')

    sbm, reaction = manualEC(sbm_toappend, reaction_toappend, inac)

    parsedRXNs, parsedSBM = iterate(reaction, sbm)

    s = pd.concat([sbm_og, parsedSBM])
    r = pd.concat([reaction_og, parsedRXNs])

    r, s = clean_RXN(r, s)

    r.to_csv(path+'/Files/appended_Reaction.csv', index=False)
    s.to_csv(path+'/Files/appended_SpeciesBaseMechanism.csv', index=False)

if __name__ == '__main__':
    main()
