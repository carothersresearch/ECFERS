from equilibrator_api import ComponentContribution, Reaction
import numpy as np
import json
import gzip
import os
import pickle

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
                try:
                    with open(os.getcwd()+'/src/thermo_calculations/thermo_cache.pickle', 'rb') as handle:
                        thermo_cache = pickle.load(handle)
                except:
                    thermo_cache = {}
                
                if r in thermo_cache:
                    rxn = thermo_cache[r]
                else:
                    rxn = Reaction({cc.get_compound("kegg:"+species[1]):species[0] for species in RXNs[r]})
                    thermo_cache[r] = rxn
                    with open(os.getcwd()+'/src/thermo_calculations/thermo_cache.pickle', 'wb') as handle:
                        pickle.dump(thermo_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

                if not rxn.is_balanced():
                    try:
                        rxn = cc.balance_by_oxidation(rxn)
                    except Exception as e:
                        print(e)
                        pass
                keqs.append({'value':np.exp((-cc.standard_dg_prime(rxn)/cc.RT).value).magnitude,'error':np.exp((-cc.standard_dg_prime(rxn)/cc.RT).error).magnitude})
            except:
                pass
    except:
        pass
    return keqs

def keq_from_kegg(reaction_kegg):
    try:
        with open(os.getcwd()+'/src/thermo_calculations/thermo_cache.pickle', 'rb') as handle:
            thermo_cache = pickle.load(handle)
    except:
        thermo_cache = {}
    
    if reaction_kegg in thermo_cache:
        return thermo_cache[reaction_kegg]
    else:
        try:
            rxn = Reaction({cc.get_compound("kegg:"+species[1]):species[0] for species in RXNs[reaction_kegg]})

            if not rxn.is_balanced():
                try:
                    rxn = cc.balance_by_oxidation(rxn)
                except Exception as e:
                    print(e)
                    pass
            keq = {'value':np.exp((-cc.standard_dg_prime(rxn)/cc.RT).value).magnitude,'error':np.exp((-cc.standard_dg_prime(rxn)/cc.RT).error).magnitude}
            thermo_cache[reaction_kegg] = keq
            with open(os.getcwd()+'/src/thermo_calculations/thermo_cache.pickle', 'wb') as handle:
                pickle.dump(thermo_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
                print(e)
                keq = None
    return keq

# sp = [c.inchi_key for c in list(rxn.sparse_with_phases.keys())]
