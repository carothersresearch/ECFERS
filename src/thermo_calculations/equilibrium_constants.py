from equilibrator_api import ComponentContribution
import numpy as np

cc = ComponentContribution()
reaction_string = "L-serine = pyruvate + NH3"
rxn = cc.search_reaction(reaction_string)
sp = [c.inchi_key for c in list(rxn.sparse_with_phases.keys())]

Keq = np.exp((-cc.standard_dg_prime(rxn)/cc.RT).value)