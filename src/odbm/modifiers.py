from overrides import overrides, final
from odbm.odbm_main import Mechanism
import numpy as np
import re

class Modifier(Mechanism):
    """
    A superclass class used to handle basic Mechansim modification functionality. Inherits from Mechanism.
    Other mechanism should inherint this class and override attributes and apply(rxn_rate)

    Attributes
    ----------
    name : str
        label used to identify mechanism
    required_params : list
        list with parameter strings, default []
    nS : int
        number of required substrates, default np.nan
    nC : int
        number of required cofactors, default np.nan
    nP : int
        number of required products, default np.nan
    nE : int
        number of required enzymes, default np.nan

    Methods
    -------
    apply(rxn_rate: str):
        Apply modification to reaction rate string

    """

    name = 'base_modifier'  # name for the mechanism
    required_params = []    # list of required parameters

    def __init__(self, rxn):
        super().__init__(rxn)
        super().writeEquation()

    @overrides
    @final
    def writeEquation(self) -> str:
        return

    def apply(self, rxn_rate: str) -> str:
        """
        Apply modification to reaction rate string

        Parameters
        ----------
        rxn_rate : str
            Original reaction rate

        Returns
        -------
        str
            Modified reaction rate
        """
        return

class Inhibition(Modifier):
    name = 'base_inhibition'

    def alpha(self, a, I, Ki) -> str:
        return a+' = (1 + '+I+'/'+Ki+')'
    
    def competitive(self, var: str, a: str): # change just Km
        mod = a+'*'+var
        return var, mod

    def noncompetitive(self, var: str, a: str): # change just kcat
        mod = '('+var+'/'+a+')'
        return var, mod

    def uncompetitive(self, vars: list, a: str): # change both kcat and Km
        mods = []
        for v in vars:
            mods.append('('+v+'/'+a+')')
        return vars, mods

    # for mixed inhibition just call competitive and uncompetitive

    def linear(self, var: str, C: str, maxC: str):
        mod = var+' * ('+C+'/'+maxC+')'
        return var, mod

    def inverse_linear(self, var: str, C: str, maxC: str):
        mod = var+' * (1-'+C+'/'+maxC+')*piecewise(1, '+C+'<'+ maxC+', 0)'
        return var, mod 

class ProductInhibition(Inhibition):
    name = 'PI'
    required_params = ['KiP.+'] # regex to accept multiple. how to specifify which product is affecting which substrate?
    nP = np.nan # or error if 1 ...

    @overrides
    def apply(self, rxn_rate: str) -> str:
        # P = [p for p in self.params.keys() if re.match(self.required_params[0], p)]
        for p in self.relevent_params:
            id = p[-1]
            a = 'a'+id+'_'+self.label
            Ki = p+'_'+self.label
            I = self.products[int(id)]
            
            Km = 'Km' + id # assuming 1st product inhibits 1st substrate !
            Km, aKm = self.competitive(Km, a)

            rxn_rate = rxn_rate.replace(Km, aKm)
            rxn_rate += '; ' + self.alpha(a, I, Ki)

        return rxn_rate

class SimpleProductInhibition(Inhibition):
    name = 'SPI'
    required_params = ['maxC.+']

    @overrides
    def apply(self, rxn_rate) -> str:
        # P = [p for p in self.params.keys() if re.match(self.required_params[0], p)]
        for p in self.relevent_params:
            id = int(p[-1])
            C = self.products[id]
            maxC, = [p+'_'+self.label if '$' not in p else p.replace('$','_') for p in [p]]

            rxn_rate = self.inverse_linear(rxn_rate, C, maxC)[1]

        return rxn_rate
    
class LinearCofactor(Inhibition):
    name = 'LC'                                     
    required_params = ['maxC.+']                     
    nC = 1

    @overrides
    def apply(self, rxn_rate) -> str:
        # P = [p for p in self.params.keys() if re.match(self.required_params[0], p)]
        for p in self.relevent_params:
            id = int(p[-1])
            C = self.cofactors[id]
            maxC, = [p+'_'+self.label if '$' not in p else p.replace('$','_') for p in [p]]

            rxn_rate = self.linear(rxn_rate, C, maxC)[1]

        return rxn_rate

class HillCofactor(Modifier):
    name = 'HC'
    required_params = ['Ka','n']
    nC = 1

    @overrides
    def apply(self, rxn_rate: str) -> str:
        C = self.cofactors[0]  # what if there are multiple cofactors? 
        Ka,n = [p+'_'+self.label for p in self.required_params]

        return rxn_rate+' * (1/(1+('+Ka+'/'+C+')^'+n+'))'  # could include this in Inhibition