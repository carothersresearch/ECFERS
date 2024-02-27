import pandas as pd
import itertools
import tellurium as te

from src.odbm.utils import extractParams, fmt

from src.odbm.mechanisms import *

# from src.parameter_balancing.balancer import RRBalancer

class ModelBuilder:
    """
    A class used to keep species and reaction information and compile them into an Antaimony model

    Attributes
    ----------
    mech_dict : str
        a dictionary with available Mechanisms
    species : pd.DataFrame
        dataframe where each row is a different species
    rxns : pd.DataFrame
        dataframe where each row is a different reaction

    Methods
    -------
    addMechanism(self, new_mechanism: Mechanism):
        Adds a new Mechanism to the internal mechanism dictionary 

    addSpecies(self, Label, StartingConc, Type = np.nan, Mechanism = np.nan, Parameters = np.nan):
        Adds a new species to the internal species dataframe 

    addReaction(self, Mechanism, Substrate, Product, Parameters, Enzyme = np.nan, Cofactor = np.nan, Label = np.nan):
        Adds a new reaction to the internal reaction dataframe
    
    applyMechanism(self, mechanism, species):
        Adds TX or TL reaction to dataframe

    writeSpecies(self, rxn):
        Write string for species initialization

    writeReaction(self, rxn, equation = True):
        Writes string for reaction definition

    writeParameters(self, parameters, label):
        Write string for parameter initialization

    get_substrates(self, id: int or str, cofactors = True):
        Returns a list of susbtrates given a reaction index
    
    get_products(self, id: int or str):
        Returns a list of products given a reaction index
    
    compile():
        Iterates through all species and reactions and generates an Antimony string
    
    saveModel(self, filename:str):
        Saves the Antimony model to a text file

    """

    def __init__(self, species, reactions):
        self.mech_dict = {}
        [self.addMechanism(m) for m in MECHANISMS]
        self.species = species
        self.rxns = reactions

    def addMechanism(self, new_mechanism: Mechanism):
        """Adds a new Mechanism to the internal mechanism dictionary 

        Parameters
        ----------
        new_mechanism (Mechanism): Mechanism class 
        """
        self.mech_dict[new_mechanism.name] = new_mechanism

    def addSpecies(self, Label, StartingConc, Type = np.nan, Mechanism = np.nan, Parameters = np.nan):
        """
        Adds a new species to the internal species dataframe

        Parameters
        ----------
        Label : str
        StartingConc : str
        Type : str, optional, by default np.nan
        Mechanism : str, optional, by default np.nan
        Parameters : str, optional, by default np.nan
        """
        args = locals()
        args.pop('self')
                # maybe check inputs??

        if not self.species['Label'].str.contains(Label).any(): # if this species does not already exist
            self.species = self.species.append(args,ignore_index = True) 
        else:
            raise('This species already exists in dataframe.')

    def addReaction(self, Mechanism, Substrate, Product, Parameters, Enzyme = np.nan, Cofactor = np.nan, Label = np.nan):
        """
        Adds a new reactions to the internal reaction dataframe

        Parameters
        ----------
        Mechanism : str
        Substrate : str
        Product : str
        Parameters : str
        Enzyme : str, optional, by default np.nan
        Cofactor : str, optional, by default np.nan
        Label : str, optional, by default np.nan
        """
        args = locals()
        args.pop('self')
        # maybe check inputs??
        # maybe do something about the Label        
        self.rxns = self.rxns.append(args,ignore_index = True)

    def applyMechanism(self, mechanism, species, function = False):
        """[summary]

        Args:
            mechanism (str): label for mechanism 
            species (str): species names

        Returns:
            None
        """        
        M = self.mech_dict[mechanism]
        substrate = fmt(species['Label'])
        label = M.generate_label(substrate)
        self.species[self.species['Label'] == species['Label']]=label              
        parameters = species['Parameters']
        pdict = extractParams(parameters)

        def lookup(lbl:str):
            K = '0'
            for k in pdict.keys():
                if lbl in k:
                    K = pdict[k]
            return K

        if M.nS > 1:
            substrate = substrate +';'+ M.required_substrates
            for s in M.required_substrates.split(';'):
                self.addSpecies(s, lookup(s))

        if not np.isnan(M.nE):
            enzyme = M.required_enzyme
            for e in enzyme.split(';'):
                self.addSpecies(e, lookup(e))
        else:
            enzyme = np.nan

        if not np.isnan(M.nC):
            cofactor = M.required_cofactor
            for c in cofactor.split(';'):
                self.addSpecies(c, lookup(c))
        else:
            cofactor = np.nan
        
        if not function: 
            product = M.generate_product(substrate)
            for p in product.split(';'):
                self.addSpecies(p, lookup(p))

            self.addReaction(mechanism, substrate, product, parameters, enzyme, cofactor, Label = label)

        else:
            return M.writeFun(substrate, M.required_params, label)

    def writeSpecies(self, species):
        """Write string for species initialization

        Args:
            species (dict): contains Label, StartingConc

        Returns:
            str: initialized species
        """        
        if species['Type'] == 'Enzyme':
            label = 'EC'+species['EC'].replace('.','')
            present = 'p_'+label+'*'
        else:
            label = fmt(species['Label'])
            present = ''
        species['Label'] = label
        relative = species['Relative']
        
        if not pd.isnull(relative):
            s_str = (label +'= ' + present + relative + '*'+str(species['StartingConc']) + '*dilution_factor; \n')
        else:
            s_str = (label +'=' + present + str(species['StartingConc']) + '*dilution_factor; \n')
        
        if not pd.isnull(species['Conc']):
                funs = species['Conc'].split(';')
                for f in funs:
                        s_str += self.applyMechanism(f,species, True)+'; \n' # not mechanism, just function   

        return s_str

    def writeReaction(self, rxn, equation = True):
        """Writes string for reaction definition

        Args:
            rxn (dict): contains species, products, mechanism, parameters

        Raises:
            KeyError: No mechanism found with that name

        Returns:
            str: reaction definition
        """        
        try:
            m = rxn['Mechanism'].split(';')
        except:
            m = ['MRL'] # defalt to modular rate law

        try:
            M = self.mech_dict[m[0].strip()](rxn)
        except KeyError:
            # bug here: throws error for no mechanism found even if issue is incorrect parameters
            raise KeyError('No mechanism found called '+m[0])
        
        if equation: eq_str = M.writeEquation()+'; \n'
        else: eq_str = ''

        rate_str = M.writeRate()
        for mod in m[1:]:
            MOD = self.mech_dict[mod.strip()](rxn)
            rate_str = MOD.apply(rate_str)

        return '\n' + eq_str + rate_str+'; '

    def writeParameters(self, parameters, label, required = True):
        """Write string for parameter initialization
        Args:
            parameters
            label
            required

        Returns:
            str: initialized parameters
        """        
        p_str = ''
        if not pd.isnull(parameters):
            #initialize value
            pdict = extractParams(parameters)
            for key, value in pdict.items():

                if '$' in key:
                    key = key.replace('$','_')
                else:
                    key = key+'_'+label

                if key not in self.p_str:
                    p_str += (key +'=' + str(value) + '; \n')
        else:
            if required:
                raise('No parameters found for reaction '+label)
            else:
                pass
            # Diego: what about default parameters? say if we want to set all transcription rates to be the same

        if len(p_str)>0:p_str =p_str+'\n'
        return p_str
    
    def writeVariable(self, variable, value = 1):
        """Write string for variable initialization
        Args:
            variable, unique string
            value, default 1

        Returns:
            str: initialized variable
        """
        v_str = ''
        if not pd.isnull(variable):
            if variable not in self.v_str:
                v_str += (variable +'=' + str(value) + '; \n')

        return v_str
    
    def compile(self) -> str:
        """
        Iterates through all species and reactions and generates an Antimony string

        Returns
        -------
        str
            Antimony model string
        """
        
        self.s_str = '# Initialize concentrations \n'
        self.p_str = '\n# Initialize parameters \n'
        self.v_str = '\n# Initialize variables \n'
        self.r_str = '# Define specified reactions \n'

        S = self.species.copy()
        for _,s in S.iterrows():
            if not pd.isnull(s['Mechanisms']):
                mechanisms = s['Mechanisms'].split(';')
                for m in mechanisms:
                        self.applyMechanism(m,s)

        for _, sp in self.species.iterrows():
            self.s_str += self.writeSpecies(sp)
            self.s_str += self.writeParameters(sp['Parameters'], sp['Label'], required = False)
            self.v_str += self.writeVariable(sp['Relative'])
            if sp['Type'] == 'Enzyme':
                self.v_str += self.writeVariable('p_'+'EC'+sp['EC'].replace('.',''))
        self.v_str += self.writeVariable('dilution_factor')

        for _, rxn in self.rxns.iterrows():
            try:
                parameters = rxn['Parameters']
            except:
                ki = ';' + rxn['KI'] if not pd.isnull(rxn['KI']) else ''
                parameters = rxn['Km'] + '; ' + rxn['Kcat'] + ki
            self.p_str += self.writeParameters(parameters, rxn['Label'])
            self.r_str += self.writeReaction(rxn) + '\n'

            if not pd.isnull(rxn['KI']):
                for i in rxn['Inhibitors'].split(';'): 
                    var = 'Gi_'+i+'_'+rxn['Label']
                    self.v_str += self.writeVariable(var, value = 0.5)

        ## hack for now to get parameters
        # flag = True
        # while flag:
        #     try:
        #         te.loada(self.s_str + self.p_str + self.r_str)
        #         flag = False
        #     except Exception as e:
        #         self.p_str += str(e).split("'")[1] +'= 1 ; \n'

        return self.s_str + self.p_str + self.v_str + self.r_str

    def saveModel(self, filename:str):
        """
        Saves the Antimony model to a text file

        Parameters
        ----------
        filename : str
        """
        with open(filename, 'w') as f:
            f.write(self.compile())

    def get_reaction(self, id):
        if type(id) is int:
            r = self.rxns.iloc[id]
        elif type(id) is str:
            r = self.rxns[self.rxns['Label'] == id]
        return r
    
    def get_substrates(self, id: int or str, cofactors = True) -> list:
        """
        Returns a list of susbtrates given a reaction index

        Parameters
        ----------
        id : int or str
            Reaction number or label
        cofactors : bool, optional
            Also return cofactors, by default True

        Returns
        -------
        List
        """
        r = self.get_reaction(id)

        if cofactors and (str(r['Cofactor']) != 'nan'):
            X = [*r['Substrate'].split(';'), *r['Cofactor'].split(';')]
        else:
            X = r['Substrate'].split(';')

        return list(map(fmt, X))

    def get_products(self, id: int or str) -> list:
        """
        Returns a list of products given a reaction index

        Parameters
        ----------
        id : int or str
            Reaction number or label

        Returns
        -------
        List
        """
        r = self.get_reaction(id)
        return list(map(fmt, r['Product'].split(';')))

    def balance(self, thermodynamics = True):
        self.balancer = RRBalancer(self)
        if thermodynamics: self.balancer.add_thermodynamics()
        balanced_parameters = self.balancer.balance(data = list(self.balancer.data.values()))
        # TODO: update parameters in antimony 
        return balanced_parameters

class ModelHandler:
    def __init__(self, model) -> None:
        self.model = model
        self.ParameterScan = {}
        self.SimParams = {}
        self._updateModel(model)
    
    def _updateModel(self, model):
        self.rr = te.loada(model)

        try: 
            self.setParameterScan(self.ParameterScan)

        except Exception as e:
            print('Could not set old parameter scan for new model\n')
            print(e)
            self.newModel_flag = True

    def setParameterScan(self, parameters_dict: dict):
        if np.all([p in self.rr.getGlobalParameterIds()+
                            self.rr.getDependentFloatingSpeciesIds()+
                                self.rr.getIndependentFloatingSpeciesIds() for p in parameters_dict.keys()]):

            if np.all([iter(v) for v in parameters_dict.values()]):
                self.ParameterScan = parameters_dict
                self.newModel_flag = False

            else:
                raise Exception('Not iterable')
        else:
            raise Exception('No parameter found')
        

    def setBoundarySpecies(self, species_dict):
        # needs to re-load the model!
        for old_s, new_s in species_dict.items():
            self.model = self.model.replace(old_s, new_s, 1)

        self._updateModel(self.model)

    def setSimParams(self,start,end,points,selections):
        self.SimParams['start'] = start
        self.SimParams['end'] = end
        self.SimParams['points'] = points
        self.SimParams['selections'] = selections

    def sensitivityAnalysis(self, metrics = []):
        if not self.SimParams:
            print('Need to specify simulation parameters')
            return
        if self.newModel_flag:
            print('A new model was loaded, no parameter scan has been specified')
            return

        self.conditions = np.array(list(self.ParameterScan.values())).T
        parameters = self.ParameterScan.keys()
        results = [None]*len(self.conditions)

        if metrics:
            results_metrics = np.empty(shape = (len(self.conditions), len(metrics)))
        
        for k,c in enumerate(self.conditions):
            self.rr.resetAll()

            for p,v in self.ConstantParams.items():
                self.rr[p]=v

            for p,v in zip(parameters,c):
                self.rr[p]=v

            try:
                sol = self.rr.simulate(self.SimParams['start'],self.SimParams['end'],self.SimParams['points'],self.SimParams['selections'])
                for j,m in enumerate(metrics):  # compare efficiency to stacking results and doing vector
                    results_metrics[k,j] = m(sol)

                results[k] = sol
            except Exception as e:
                print(e)

        if metrics:
            return results, results_metrics
        else:
            return results