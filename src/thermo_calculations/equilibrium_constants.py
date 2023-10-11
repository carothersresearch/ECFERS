from equilibrator_api import ComponentContribution, Reaction
import numpy as np
import json
import gzip
import os

path = os.getcwd()+'/src/thermo_calculations'

with gzip.open(path+"/kegg_enzymes.json.gz", "r") as f:
        ECs = {e['EC']:e['reaction_ids'] for e in json.load(f)}

with gzip.open(path+"/kegg_reactions.json.gz", "r") as f:
        RXNs = {r['RID']:r['reaction'] for r in json.load(f)}

cc = ComponentContribution()

def keq_from_ec(ec_string):
    keqs = []
    try:
        for r in ECs[ec_string]:
            try:
                rxn = Reaction({cc.get_compound("kegg:"+species[1]):species[0] for species in RXNs[r]})
                if not rxn.is_balanced():
                    rxn = cc.balance_by_oxidation(rxn)
                keqs.append({'value':np.exp((-cc.standard_dg_prime(rxn)/cc.RT).value).magnitude,'error':np.exp((-cc.standard_dg_prime(rxn)/cc.RT).error).magnitude})
            except:
                pass
    except:
        pass
    return keqs

# sp = [c.inchi_key for c in list(rxn.sparse_with_phases.keys())]
