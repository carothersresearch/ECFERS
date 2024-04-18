import os
import tellurium as te
import pandas as pd
import sys
import os

# load SBML model
modelfile = os.getcwd()+'/240226MC_FULL.sbml'
#modelfile = sys.argv[1]
print(modelfile)
r = te.loadSBMLModel(modelfile)


reaction = pd.read_csv(sys.argv[2])
sbm = pd.read_csv(sys.argv[3])

rxn = reaction.copy()
s = sbm.copy()

for index, row in s.iterrows():
    if row['Type'] == 'Metabolite':
        KEGG = row['Label']
        try:
            newCompConc = r[f'init([{KEGG}])']/r['dilution_factor']
        except RuntimeError:
            newCompConc = row['StartingConc']
            # print(f'ID {KEGG} is not found in the SBML model. Using initial concentration provided in SpeciesBaseMechanism.')
        s.loc[index, 'StartingConc'] = newCompConc
    if row['Type'] == 'Enzyme':
        ECjoined = 'EC'+''.join(row['EC'].split('.'))
        pEC = 'p_' + ECjoined
        try:
            newEnzConc = r[f'init([{ECjoined}])']/(r['p_' + ECjoined]*r['dilution_factor'])
        except RuntimeError:
            newEnzConc = row['StartingConc']
        s.loc[index, 'StartingConc'] == newEnzConc


for index, row in rxn.iterrows():
    ECjoined = 'EC'+''.join(row['EC'].split('.'))
    label = row['Label']
    compounds = ','.join([row['Substrates'], row['Products']])
    inhibitors = row['Inhibitors']

    km_list = []
    ki_list = []

    for compound in compounds.split(','):
        new_KM = r[(f'Km_{compound}_{label}')]
        km_list.append(f'Km_{compound}: {new_KM}')
    rxn.loc[index, 'Km'] == (';').join(km_list)

    kcf = r[(f'Kcat_F_{label}')]
    kcr = r[(f'Kcat_R_{label}')]
    rxn.loc[index, 'Kcat'] == f'Kcat_F: {kcf}; Kcat_R: {kcr}'

    for inhibitor in inhibitors.split(';'):
        new_KI = r[(f'Ki_{compound}_{label}')]
        ki_list.append(f'{compound}_KI: {new_KI}')
    rxn.loc[index, 'KI'] == (';').join(ki_list)

rxn.to_csv('FittedReaction.csv', index=False)
s.to_csv('FittedSpeciesBaseMechanisms.csv', index=False)
