from equilibrator_api import ComponentContribution

cc = ComponentContribution()
reaction_string = "L-serine = pyruvate + NH3"
rxn = cc.search_reaction(reaction_string)
sp = [c.inchi_key for c in list(rxn.sparse_with_phases.keys())]

cc.physiological_dg_prime(rxn)