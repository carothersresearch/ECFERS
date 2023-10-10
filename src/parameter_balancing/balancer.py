
from symbolicSBML import SBMLModel, Parameters
from parameter_balancer import Balancer, ParameterData
import numpy as np


class RRBalancer():

    # Priors are linear scale here and will be converted inside.
    default_priors = {
        # Base quantities
        Parameters.mu: (-880.0, 680.00),
        Parameters.kv: (10.0, 6.26),
        Parameters.km: (0.1, 6.26),
        Parameters.c: (0.1, 10.32), # we might want to change this
        Parameters.u: (0.0001, 10.32), # we might want to change this
        Parameters.ki: (0.1, 6.26),
        Parameters.ka: (0.1, 6.26),
        # Derived quantities
        Parameters.keq: (1.0, 10.32),
        Parameters.kcat_prod: (10.0, 10.32),
        Parameters.kcat_sub: (10.0, 10.32),
        Parameters.vmax: (0.001, 17.01),
        Parameters.A: (0.0, 10.00),
        Parameters.mu_p: (-880.0, 680.00),
    }

    default_data = ParameterData([], [], [], [], [])

    def __init__(self, rr)
        self.rr = rr
        self._get_metabolites()
        self._get_reactions()
        self._get_stoichiometry()
        self.structure = SBMLModel.from_structure(self.metabolites, self.reactions, self.stoichiometry)

    def balance(self, priors = default_priors, data = default_data, T = 300, R = 8.314 / 1000.)
        # maybe all data needs to be priors 
        self.balanced_parameters = Balancer(priors, data, self.structure, T=T, R=R, augment=True).balance(sparse=False).to_frame(mean=True).T
        return self.balanced_parameters
    
    def _get_metabolites(self):
        self.metabolites = self.rr.model.getFloatingSpeciesIds()
        return

    def _get_reactions(self):
        self.reactions = self.rr.model.getReactionIds()
        return

    def _get_stoichiometry(self):
        self.stoichiometry = np.array(self.rr.getFullStoichiometryMatrix())
        return

    def _parse_parameters(self):
        parameters = self.rr.model.getGlobalParameterIds() # Kms, kcats, and other
        p_values = self.rr.model.getGlobalParameterValues()
        # r.model.getFloatingSpeciesInitConcentrationIds()
        
        c_values = self.rr.model.getFloatingSpeciesInitConcentrations()
        # for each metabolite
        # (Parameters.c, 'G6P', None, 10.0, 1.0)

        # for each enzyme/reaction 
        # (Parameters.u, None, 'R1', 10.0, 1.0)

        return

    def add_thermodynamics(self, keqs=None):

        if keqs:
            #add them
        else:
        # get equilibrium data w eQuilibrator

data = [
    (Parameters.c, 'G6P', None, 10.0, 1.0),
    (Parameters.c, 'F6P', None, 10.0, 1.0),
    (Parameters.km, 'G6P', 'PGI', 0.28, 0.056),
    (Parameters.km, 'F6P', 'PGI', 0.147, 0.0294),
    (Parameters.keq, None, 'PGI', 0.361, 0.0361),
    (Parameters.vmax, None, 'PGI', 1511, 151),
]

data_r = ParameterData(*zip(*data))


