from equilibrator_api import ComponentContribution, Reaction
import numpy as np
import json
import gzip
import pandas as pd

with gzip.open("kegg_enzymes.json.gz", "r") as f:
        ECs = {e['EC']:e['reaction_ids'] for e in json.load(f)}

with gzip.open("kegg_reactions.json.gz", "r") as f:
        RXNs = {r['RID']:r['reaction'] for r in json.load(f)}

cc = ComponentContribution()
list_of_ECs = [list(ECs.keys())[i] for i in [5,6,10,12,23]]
model_ECs = pd.read_csv('Reaction.csv', usecols=['EC'])

db = pd.DataFrame(columns=['EC','KEGG Rid','R string','Keq'])
for ec_string in model_ECs:
    try:
        for r in ECs[ec_string]:
            print(r)
            try:
                rxn = Reaction({cc.get_compound("kegg:"+species[1]):species[0] for species in RXNs[r]}) # could make this faster by reusing compounds
                if not rxn.is_balanced():
                    rxn = cc.balance_by_oxidation(rxn)
                r_string = rxn.__str__()
                Keq = np.exp((-cc.standard_dg_prime(rxn)/cc.RT).value)

            except:
                r_string = ''
                Keq = np.nan
    except:
        r = ''
        r_string = ''
        Keq = np.nan

    db = pd.concat([pd.DataFrame([[ec_string,r,r_string,Keq]], columns=db.columns), db], ignore_index=True)

print(db)
# rxn = cc.search_reaction(reaction_string)
# sp = [c.inchi_key for c in list(rxn.sparse_with_phases.keys())]
# print(cc.standard_dg_prime(rxn))
# Keq = np.exp((-cc.standard_dg_prime(rxn)/cc.RT).value)
# print(Keq)