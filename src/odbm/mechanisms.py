from overrides import EnforceOverrides, overrides, final
from src.odbm.utils import extractParams, fmt, getStoich
import pandas as pd
import numpy as np
import re

class InputError(Exception):
    pass

class Mechanism(EnforceOverrides):
    """
    A superclass class used to handle basic Mechansim functionality: format inputs and write equation.
    Other mechanism should inherint this class and override attributes and writeRate()

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
    writeEquation():
        Writes chemical equations in form of 'S + E → E + P'

    writeRate():
        Writes rate of chemical reaction. This function should always be overriden.

    """

    # these variables should be overriden in new mechanisms
    name = 'base_mechanism'
    required_params = []
    nS = np.nan                  
    nC = np.nan
    nP = np.nan            
    nE = np.nan

    def __init__(self,rxn: pd.DataFrame):

        try:
            self.enzyme = 'EC'+rxn['EC'].replace('.','')
            self.substrates = rxn['Substrates']
            self.products = rxn['Products']
            self.inhibitors = rxn['Inhibitors']
            try:
                self.inhibitors = ';'.join([I for I in np.unique(self.inhibitors.split(';')) if np.all([i not in I for i in ['D','G']])])
            except:
                self.inhibitors = 'nan'
            #self.cofactors = rxn['Cofactor'] # this is going to be removed. maybe we will add activators/inhibitors
            # try:
            #     self.params = rxn['Parameters'] # this is going to be two (or more?) columns
            # except:
            self.params = rxn['Km'] + '; ' + rxn['Kcat']
            # try:
            #     self.label = rxn['Reaction ID'] # it was confusing to have Reaction ID and Label. sticking with just label
            # except:
            self.Ki = rxn['KI']
            try:
                self.Ki = ';'.join([I for I in np.unique(self.Ki.split(';')) if np.all([i not in I for i in ['D','G']])])
            except:
                self.Ki = np.nan

            self.label = rxn['Label']
            self.EC = rxn['EC']

        except:
            raise KeyError("Missing Reaction fields")

        self._processInput()
        self._formatInput()
    
    @final
    def _processInput(self):
        """
        Checks user model definition for errors.

        Raises:
            InputError: if missing required kinetic parameter for specified mechanism type
            InputError: if an incorrect number of enzymes, cofactors, substrates, or products are given 
                        for a specific mechanism type

        """
        # params
        self.params = extractParams(self.params)
        if self.required_params:

            self.relevent_params = sum([[P for P in self.params if re.match(p,P) ] for p in self.required_params],[])

            if not np.all([np.any([re.match(p,P) for P in self.params]) for p in self.required_params]):
                raise InputError("No "+' or '.join(self.required_params)+" found in parameters for reaction "+self.label)

        # # cofactor
        # if str(self.cofactors) != 'nan':
        #     self.cofactors = self.cofactors.split(';')
        # else:
        #     self.cofactors = []
        # if len(self.cofactors) != self.nC and np.isnan(self.nC) == False:
        #     raise InputError(str(len(self.cofactors))+' cofactor(s) found for a '+ str(self.nC) + ' cofactor mechanism in reaction '+self.label)

        # enzyme
        if str(self.enzyme) != 'nan':
            self.enzyme = self.enzyme.split(';')
        else:
            self.enzyme = []
        if len(self.enzyme) != self.nE and np.isnan(self.nE) == False:
            raise InputError(str(len(self.enzyme))+' enzyme(s) found for a '+ str(self.nE) + ' enzyme mechanism in reaction '+self.label)

        # substrates
        if str(self.substrates) != 'nan':
            self.substrates = self.substrates.split(';')
        else:
            self.substrates = []
        if len(self.substrates) != self.nS and np.isnan(self.nS) == False:
            raise InputError(str(len(self.substrates))+' substrate(s) found for a '+ str(self.nS) + ' substrate mechanism in reaction '+self.label)

        # products 
        self.products = self.products.split(';')
        if (not np.isnan(self.nP)) and (len(self.products) != self.nP):
            raise InputError(str(len(self.products))+' product(s) ('+str(self.products)+') found for a '+ str(self.nP) + ' product mechanism in reaction '+self.label)
    
        # inhibitors
        if str(self.inhibitors) != 'nan':
            self.inhibitors = self.inhibitors.split(';')
        else:
            self.inhibitors = []

    @final
    def _formatInput(self):
        #calls fmt function in utils to format input strings to be antimony compatible

        # if using kegg ids probably not needed!

        self.products = list(map(fmt, self.products))
        self.substrates = list(map(fmt, self.substrates))
        self.inhibitors = list(map(fmt, self.inhibitors)) if not pd.isnull(self.Ki) else self.inhibitors
        self.enzyme = list(map(fmt, self.enzyme))
        #self.cofactors = list(map(fmt, self.cofactors))

    def writeEquation(self) -> str:
        """
        Writes chemical equations in form of S + E → E + P

        Returns: 
        -------
        rxn_str (str) reaction equation in string format
        """
        
        allS = ' + '.join(self.substrates)
        allE = ' + '.join(self.enzyme)
        allP = ' + '.join(self.products)

        if self.enzyme != 'nan' and self.enzyme != []:
            rxn_str = allS + ' + ' + allE + ' -> ' + allE + ' + '  + allP
        else: 
            rxn_str = allS + ' -> ' + allP

        self.stoich = [getStoich(s)[0] for s in self.substrates]
        self.substrates = [getStoich(s)[1] for s in self.substrates]
        self.products = [getStoich(s)[1] for s in self.products]

        return self.label +' : '+rxn_str
    
    def writeRate(self) -> str:
        """
        Writes rate of chemical reaction. This function should always be overriden.

        Returns
        -------
        str
        """
        pass

class MichaelisMenten(Mechanism):
    name = 'MM'                        # name for the mechanism
    required_params = ['kcat','Km']    # list of required parameters
    nS = 1                             # number of required substrates 
    nP = np.nan                        # number of required products 
    nE = 1                             # enzymatic reaction

    @overrides
    def writeRate(self) -> str:
        S = self.substrates
        E = self.enzyme[0]
        kcat,Km = [p+'_'+self.label for p in self.relevent_params]

        return self.label +' = '+ kcat + '*'+E+'*'+S[0]+'/('+Km+' + '+S[0]+')'

class ModularRateLaw(Mechanism):
    """
    Based on: https://academic.oup.com/bioinformatics/article/26/12/1528/281177#393582847
    """
    name = 'MRL'
    required_params = None # could check some other ways but skipping check for now
    nS = np.nan                        # number of required substrates 
    nP = np.nan                        # number of required products 
    nI = np.nan                        # number of required inhibitors
    nE = 1
    ignore = ['C00001','C00080'] # H2O, H+   

    def _ignore_species(self, species):
        return [s for s in species if s not in self.ignore]

    def numerators(self) -> str:
        substrates = self._ignore_species(self.substrates)
        products = self._ignore_species(self.products)

        allS = '*'.join(substrates)
        allP = '*'.join(products)

        allKmS = '*'.join(['Km_'+s+'_'+self.enzyme[0] for s in substrates])
        allKmP = '*'.join(['Km_'+p+'_'+self.enzyme[0] for p in products])

        kcatF = 'Kcat_F_' + self.label
        kcatR = 'Kcat_R_' + self.label

        return '('+kcatF+'*('+ allS +')/('+ allKmS +'))', '('+ kcatR +'*('+ allP +')/('+ allKmP +'))'
    
    def denominator(self) -> str:
        substrates = self._ignore_species(self.substrates)
        products = self._ignore_species(self.products)

        allKmS = ['Km_'+s+'_'+self.enzyme[0] for s in substrates]
        allKmP = ['Km_'+p+'_'+self.enzyme[0] for p in products]
        allS = '*'.join(['(1+'+s+'/'+Km+')' for s, Km in zip(substrates, allKmS)])
        allP = '*'.join(['(1+'+p+'/'+Km+')' for p, Km in zip(products, allKmP)])
        return '('+ allS + ' + '+ allP + ' -1)'
    
    def inhibition_nc(self) -> str: # only activity
        inhibitors = self._ignore_species(self.inhibitors)
        allKi = ['Ki_'+i+'_'+self.enzyme[0] for i in inhibitors] 
        allGnc = ['Gnc_'+i+'_'+self.enzyme[0] for i in inhibitors] # degree of inhibition (1 = no inh c, 0 = full inh)
        fr = '*'.join(['('+Gnc+'+(1-'+Gnc+')*(1/(1+'+i+'/'+Ki+')))' for i, Ki, Gnc in zip(inhibitors, allKi, allGnc)])
        return fr
    
    def inhibition_c(self) -> str: # only binding
        inhibitors = self._ignore_species(self.inhibitors)
        allKi = ['Ki_'+i+'_'+self.enzyme[0] for i in inhibitors] 
        allGc = ['Gc_'+i+'_'+self.enzyme[0] for i in inhibitors] # degree of inhibition (1 = no inh c, 0 = full inh)
        dreg = '+'.join(['(1-'+Gc+')*('+i+'/'+Ki+')' for i, Ki, Gc in zip(inhibitors, allKi, allGc)])
        return dreg
    
    @overrides
    def writeRate(self) -> str:
        u = self.enzyme[0]
        Tf, Tr = self.numerators()

        D = self.label +'_D := ' + self.denominator()

        if len(self.inhibitors)>0:
            fr = u +'_fr := ' + self.inhibition_nc()
            Dreg = u +'_Dreg := ' + self.inhibition_c()

            rate_f = self.label +'_f := '+ u + ' * ' + u +'_fr' + ' * ' + Tf + '/(' + self.label +'_D' + ' + ' + u +'_Dreg' + ')'
            rate_r = self.label +'_r := '+ u + ' * ' + u +'_fr' + ' * ' + Tr + '/(' + self.label +'_D' + ' + ' + u +'_Dreg' + ')'
            rate_net = self.label +' = '+self.label +'_f' + ' - ' + self.label +'_r'
            rate = fr + '; \n' + Dreg + '; \n' + D + '; \n' +  rate_f + '; \n' + rate_r + '; \n' + rate_net

        else:
            rate_f = self.label +'_f := '+ u + ' * ' + Tf + '/' + self.label +'_D'
            rate_r = self.label +'_r := '+ u + ' * ' + Tr + '/' + self.label +'_D'
            rate_net = self.label +' = '+self.label +'_f' + ' - ' + self.label +'_r'
            rate = D + '; \n' +  rate_f + '; \n' + rate_r + '; \n' + rate_net

        return rate

class OrderedBisubstrateBiproduct(Mechanism):
    # ordered bisubstrate-biproduct
    # must have two substrates and two products
    # https://iubmb.qmul.ac.uk/kinetics/ek4t6.html#p52
    # looks for kcat, Km1, Km2, K

    name = 'OBB'                                     # name for the mechanism
    required_params = ['kcat', 'Km1', 'Km2', 'K']    # list of required parameters
    nS = 2                                           # number of required substrates 
    nP = 2                                           # number of required products 
    nE = 1                                        # enzymatic reaction

    @overrides
    def writeRate(self) -> str:
        S = self.substrates
        E = self.enzyme[0]
        kcat,Km1,Km2,K = [p+'_'+self.label for p in self.relevent_params]

        return self.label +' = '+kcat+ '*'+E+'*'+(S[0])+'*'+(S[1])+'/(' \
                    +(S[0])+'*'+(S[1])+'+'+Km1+'*'+(S[1])+'+ '+Km2+'*'+(S[0])+'+'+ K+')'

class MassAction(Mechanism):
    name = 'MA'                                     # name for the mechanism
    required_params = ['k']                         # list of required parameters
    nS = np.nan                                     # number of required substrates 
    nP = np.nan                                     # number of required products 
    nE = np.nan                                      # enzymatic reaction

    # mass action kinetics
    @overrides
    def writeRate(self) -> str:
        S = self.substrates
        power = self.stoich

        k, = [p+'_'+self.label if '$' not in p else p.replace('$','_') for p in self.relevent_params]

        allS = '*'.join([s+'^'+c if len(c)>0 else s for s,c in zip(S,power)])
        allE = '*'.join(self.enzyme)
        if len(allE)>0: allE='*'+allE
        if len(allS)>0: allS ='*'+allS+allE

        return self.label +' = '+ k+allS

class MonoMassAction(Mechanism): # no superscipt
    name = 'MMA'                                     # name for the mechanism
    required_params = ['k']                         # list of required parameters
    nS = np.nan                                     # number of required substrates 
    nP = np.nan                                     # number of required products 
    nE = np.nan

    @overrides
    def writeRate(self) -> str:
        allS = '*'.join(self.substrates)
        allE = '*'.join(self.enzyme)
        if len(allE)>0: allE='*'+allE

        k, = [p+'_'+self.label if '$' not in p else p.replace('$','_') for p in self.relevent_params]
        return self.label + ' = ' + k+'*'+allS+allE

class ConstantRate(Mechanism):
    name = 'CR'
    required_parameters = ['k']

    @overrides
    def writeRate(self) -> str:
        return self.label +' = k_'+self.label

class Exponential(Mechanism):
    name = 'EXP'
    required_params = ['Cmax','tau']
    generate_label = lambda l: l

    def writeFun(var, parameters, label) -> str:
        Cmax, tau = [p+'_'+label for p in parameters]
        return var + ' := '+Cmax+'*(1-exp(-time/'+tau+'))'

class simplifiedOBB(Mechanism):
    name = 'SOBB'                                     # name for the mechanism
    required_params = ['kcat', 'Km1', 'Km2']    # list of required parameters
    nS = 2                                           # number of required substrates 
    nE = 1                                        # enzymatic reaction

    @overrides
    def writeRate(self) -> str:
        S = self.substrates
        E = self.enzyme[0]
        kcat,Km1,Km2 = [p+'_'+self.label for p in self.relevent_params]

        return self.label +' = '+ kcat + '*'+E+'*'+(S[0])+'*'+(S[1])+'/(' \
                    +(S[0])+'*'+(S[1])+'+'+Km1+'*'+(S[1])+'+'+Km2+'*'+(S[0])+'+'+Km1+ '*' +Km2+')'

# class PI(Mechanism):
class TX_MM(MichaelisMenten):
    name = 'TX_MM'
    required_enzyme = 'RNAP'
    generate_label = lambda l: l+'_TX'
    generate_product = lambda s: s + ';' + s[:-3]+'RNA'



#####################################################


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


######################################################################



MECHANISMS = [  MichaelisMenten, ModularRateLaw, OrderedBisubstrateBiproduct, MassAction, simplifiedOBB, ConstantRate, Exponential,
                        MonoMassAction, TX_MM,
                        LinearCofactor, HillCofactor, ProductInhibition, SimpleProductInhibition
                    ]