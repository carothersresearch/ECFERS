import os
from overrides import EnforceOverrides, overrides, final
import pickle

import enkie
import src.kinetic_estimator.DLKcat.model as DLKcat_model
from src.kinetic_estimator.KM_prediction_function.metabolite_preprocessing import metabolite_preprocessing
from src.kinetic_estimator.KM_prediction_function.GNN_functions import calculate_gnn_representations
from src.kinetic_estimator.KM_prediction_function.enzyme_representations import calcualte_esm1b_vectors

import torch
import requests
import json
from rdkit import Chem
import numpy as np
from collections import defaultdict
import xgboost as xgb
import esm
import shutil
import warnings
import re
import pickle

warnings.filterwarnings("ignore")

path = os.getcwd()+'/src/kinetic_estimator'

class BaseEstimator:
    """
    Example of an estimator class. Specifc estimators should inherent BaseEstimator and override main functions.
    
    """
    name = 'name of estimator'
    parameter = ['parameter(s) capable of estimating']
    _pubchem_cache = {} # probably bettter ways of doing this but it works
    _uniprot_cache = {}

    def _preprocess_inputs(self):
        """
        Makes sure inputs are in an acceptable format for estimator. This function should always be overriden.
        """
        pass

    def _load_model(self):
        """
        Writes rate of chemical reaction. This function should always be overriden.
        """
        pass

    def estimate(self, substrates:list, enzymes:list):
        """
        Method for calculating kinetic paramenter with the specfic estimator. This function should always be overriden.
        """
        pass

    # Generally useful methods:

    # One method to obtain SMILES by PubChem API using the website
    def _get_smiles(self, name):
        if name not in self._pubchem_cache:
            try :
                pubchem_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT' % name
                response = requests.get(pubchem_url)
                if response.status_code == 200:
                    # Parse the data from the response
                    smiles = response.content.splitlines()[0].decode()
                    self._pubchem_cache[name] = smiles
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}")
                    self._pubchem_cache[name] = None
            except Exception as e:
                print(f"Request failed: {e}")
                print("Could not get SMILES for "+name+"from PubChem. Returning 'None'")
                self._pubchem_cache[name] = None
        return self._pubchem_cache[name]
    
    # One method to obtain sequences by UniProt API using the website
    def _get_AAseq(self, EC:str, organism = 'Escherichia coli'):
        if EC+'_'+organism not in self._uniprot_cache:
            try:
                uniprot_url = 'https://rest.uniprot.org/uniprotkb/search?fields=sequence,protein_name&format=json&query=organism_name:"'+organism+'"+AND+ec:'+EC
                response = requests.get(uniprot_url)
                # Check if the request was successful (status code 200)
                if response.status_code == 200:
                    # Parse the JSON data from the response
                    data = json.loads(response.text)
                    AAseq = data['results'][0]['sequence']['value']
                    self._uniprot_cache[EC+'_'+organism] = AAseq
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}")
                    self._uniprot_cache[EC+'_'+organism] = None
            except Exception as e:
                print(f"Request failed: {e}")
                print("Could not get sequence string for enzyme:"+EC+" organism: "+organism+"from Uniprot. Returning 'None'")
                self._uniprot_cache[EC+'_'+organism] = None
        return self._uniprot_cache[EC+'_'+organism]
    
    def _check_str_is_prot(self, string:str) -> bool:
        return re.compile('^[acdefghiklmnpqrstvwy]*$', re.I).search(string) is not None
    
    def _check_str_is_smiles(self, string:str) -> bool:
        m = Chem.MolFromSmiles(string,sanitize=False)
        if m is None:
            return False
        else:
            try:
                Chem.SanitizeMol(m)
            except:
                print('Species string "%s" is in SMILES format but has invalid chemistry' % string)
                return False 
        return True

class Estimator:
    """
    Main Estimator class. Loads any other BaseEstimator.

    Parameters
    ----------
    estimator : str
        Name of the avaiable estimator to load
    parameter : str
        Kinetic parameter to estimate. Should match the capabilites of the estimator.
    pretrained_state : str
        Path to pretrained model. Defaults to the orgiinal pre-trained model for each estimator
    """
    def __init__(self, estimator:str, parameter:str, pretrained_state = None):
        self.estimator = None
        self.estimator_dict = {}
        [self.addEstimator(e) for e in ESTIMATORS]
        if estimator in [e.name for e in ESTIMATORS]:
            if parameter in self.estimator_dict[estimator].parameter:
                if pretrained_state:
                    self.estimator = self.estimator_dict[estimator](pretrained_state) # load user trained model
                else:
                    self.estimator = self.estimator_dict[estimator]() # defaults to original pre-trained model
            else:
                raise KeyError("Estimator " + estimator + " cannot be used to estimate " + parameter)
        else:
            raise KeyError("No estimator named " + estimator + " found. Available estimators are: " + [e.name for e in ESTIMATORS])
        
        self._load_caches()

    def _load_caches(self):
        # try loading caches
        try:
            with open(path+'/pubchem_cache.pickle', 'rb') as handle:
                self.estimator._pubchem_cach = pickle.load(handle)
        except:
            self.estimator._pubchem_cach = {}
        try:
            with open(path+'/uniprot_cache.pickle', 'rb') as handle:
                self.estimator._uniprot_cache = pickle.load(handle)
        except:
            self.estimator._uniprot_cache = {}
    
    def _dump_caches(self):
        with open(path+'/pubchem_cache.pickle', 'wb') as handle:
            pickle.dump(self.estimator._pubchem_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path+'/uniprot_cache.pickle', 'wb') as handle:
            pickle.dump(self.estimator._uniprot_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def addEstimator(self, new_estimator:BaseEstimator):
        """Adds a new Mechanism to the internal mechanism dictionary 

        Parameters
        ----------
        new_mechanism (Mechanism): Mechanism class 
        """
        self.estimator_dict[new_estimator.name] = new_estimator
    
    def estimate(self, substrates:list, enzymes:list, *args) -> float:
        #self.estimate.__doc__ = self.estimator.estimate.__doc__ # it would be nice to get this docstring from the child 
        estimates = self.estimator.estimate(substrates, enzymes, *args)
        self._dump_caches()
        return estimates
    
class ENKIE(BaseEstimator):
    name = 'ENKIE'
    parameter = ['kcat','Km']
    """
    Currently not supported.
    """
    def __init__(self) -> None:
        pass

class DLKcat(BaseEstimator):
    """
    DLKcat estimator based on 'https://github.com/SysBioChalmers/DLKcat'

    Capable of predicting kcats from substrate SMILES and enzymes sequence

    Loads pre-trained models from /DLKcat/trained_model/

    For training your own model, use original repo.

    Parameters
    ----------
    pretrained_state : str
        Path to pretrained model. Defaults to 'default_pretrained_model'
    """
    name = 'DLKcat'
    parameter = ['kcat']

    def __init__(self, pretrained_state = 'default_pretrained_model') -> None:
        super().__init__()
        self._load_model(pretrained_state)

    @overrides
    def _preprocess_inputs(self, substrate:str, enzyme:str, organism:str):
        if not self._check_str_is_smiles(substrate):
            substrate = self._get_smiles(substrate)

        if not self._check_str_is_prot(enzyme):
            enzyme = self._get_AAseq(enzyme, organism)

        return substrate, enzyme

    @overrides
    def _load_model(self, pretrained_state=None):
        # prep model
        fingerprint_dict = DLKcat_model.load_pickle(path+'/DLKcat/trained_model/fingerprint_dict.pickle')
        word_dict = DLKcat_model.load_pickle(path+'/DLKcat/trained_model/sequence_dict.pickle')
        n_fingerprint = len(fingerprint_dict)
        n_word = len(word_dict)

        dim=10
        layer_gnn=3
        window=11
        layer_cnn=3
        layer_output=3

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        self.model = DLKcat_model.KcatPrediction(self._device, n_fingerprint, n_word, 2*dim, layer_gnn, window, layer_cnn, layer_output).to(self._device)
        if pretrained_state:
            try:
                self.model.load_state_dict(torch.load(path+'/DLKcat/trained_model/'+pretrained_state, map_location=self._device))
            except:
                raise FileNotFoundError("No file named "+pretrained_state+" in "+path+"/DLKcat/trained_model/")
    @overrides
    def estimate(self, substrates: list, enzymes:list, organisms = None, full_report = False): 
        """
        Estimate kcat based on SMILES and protein sequence. If substrate is not SMILES, we will try to fetch if from PubChem. If enzymes is not a protein sequence
        we will try to fetch if from UniProt based on EC number and organism. 

        Parameters
        ----------
        substrates : list
            list of SMILES, or species names
        enzymes : list
            list of protein sequences, or EC numbers. If EC numbers, 'organims' must be specified, defaults to 'Escherichia coli'
        organisms : list, optional
            list of organisms names
        full_report : flag to return estimates, or also the process SMILES and protein sequences

        Returns
        -------
        list
            list of estimated predictions, if full_report is False
        dict
            dict of inputs, processed inputs, and estimates, if full_report is True
        """
        kcats = []
        if len(substrates) != len(enzymes): raise Exception('Mismatch in the number of substrates and enzymes')
        if not organisms: organisms = ['Escherichia coli']*len(enzymes)
        all_smiles = []
        all_seq = []
        for s, e, o in zip(substrates, enzymes, organisms):
            smiles, seq = self._preprocess_inputs(s,e,o)

            if (smiles is not None) and (seq is not None):
                try:
                    radius=2
                    ngram=3
                    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                    atoms = self._create_atoms(mol)
                    i_jbond_dict = self._create_ijbonddict(mol)
                    fingerprints = self._extract_fingerprints(atoms, i_jbond_dict, radius)
                    adjacency = self._create_adjacency(mol)
                    words = self._split_sequence(seq,ngram)

                    fingerprints = torch.LongTensor(fingerprints).to(self._device)
                    adjacency = torch.FloatTensor(adjacency).to(self._device)
                    words = torch.LongTensor(words).to(self._device)

                    kcats.append(2**self.model.forward([fingerprints, adjacency, words]).item())
                except:
                    kcats.append(None)
            else:
                kcats.append(None)
            all_smiles.append(smiles)
            all_seq.append(seq)
        if full_report:
            return {'substrates':substrates,'SMILES':all_smiles, 'enzyme':enzymes,'seqs':all_seq,'kcats':kcats}
        else: 
            return kcats
    
    def _create_atoms(self, mol):
        """Create a list of atom (e.g., hydrogen and oxygen) IDs
        considering the aromaticity."""
        atom_dict = DLKcat_model.load_pickle(path+'/DLKcat/trained_model/atom_dict.pickle')
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [atom_dict[a] for a in atoms]
        return np.array(atoms)
    def _create_ijbonddict(self, mol):
        """Create a dictionary, which each key is a node ID
        and each value is the tuples of its neighboring node
        and bond (e.g., single and double) IDs."""
        bond_dict = DLKcat_model.load_pickle(path+'/DLKcat/trained_model/bond_dict.pickle')
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict
    def _extract_fingerprints(self, atoms, i_jbond_dict, radius):
        """Extract the r-radius subgraphs (i.e., fingerprints)
        from a molecular graph using Weisfeiler-Lehman algorithm."""

        fingerprint_dict = DLKcat_model.load_pickle(path+'/DLKcat/trained_model/fingerprint_dict.pickle')
        edge_dict = DLKcat_model.load_pickle(path+'/DLKcat/trained_model/edge_dict.pickle')

        if (len(atoms) == 1) or (radius == 0):
            fingerprints = [fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                """Update each node ID considering its neighboring nodes and edges
                (i.e., r-radius subgraphs or fingerprints)."""
                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    # fingerprints.append(fingerprint_dict[fingerprint])
                    # fingerprints.append(fingerprint_dict.get(fingerprint))
                    try :
                        fingerprints.append(fingerprint_dict[fingerprint])
                    except :
                        fingerprint_dict[fingerprint] = 0
                        fingerprints.append(fingerprint_dict[fingerprint])

                nodes = fingerprints

                """Also update each edge ID considering two nodes
                on its both sides."""
                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        # edge = edge_dict[(both_side, edge)]
                        # edge = edge_dict.get((both_side, edge))
                        try :
                            edge = edge_dict[(both_side, edge)]
                        except :
                            edge_dict[(both_side, edge)] = 0
                            edge = edge_dict[(both_side, edge)]

                        _i_jedge_dict[i].append((j, edge))
                i_jedge_dict = _i_jedge_dict

        return np.array(fingerprints)
    def _create_adjacency(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency)
    def _split_sequence(self, sequence, ngram):
        sequence = '-' + sequence + '='
        word_dict = DLKcat_model.load_pickle(path+'/DLKcat/trained_model/sequence_dict.pickle')
        words = list()
        for i in range(len(sequence)-ngram+1) :
            try :
                words.append(word_dict[sequence[i:i+ngram]])
            except :
                word_dict[sequence[i:i+ngram]] = 0
                words.append(word_dict[sequence[i:i+ngram]])

        return np.array(words)

class KM_prediction(BaseEstimator):
    """
    KM_prediction estimator based on 'https://github.com/AlexanderKroll/KM_prediction_function/'

    Capable of predicting Km from substrate SMILES and enzymes sequence

    Loads pre-trained models from /KM_prediction_function/trained_model/

    For training your own model, use original repo.
    
    Parameters
    ----------
    pretrained_state : str
        Path to pretrained model. Defaults to 'xgboost_model_new_KM_esm1b'
    """
    name = 'KM_prediction'
    parameter = ['Km']

    def __init__(self, pretrained_state = 'xgboost_model_new_KM_esm1b.dat') -> None:
        super().__init__()
        self._load_model(pretrained_state)

    @overrides
    def _preprocess_inputs(self, substrate:str, enzyme:str, organism:str):
        if not self._check_str_is_smiles(substrate):
            substrate = self._get_smiles(substrate)

        if not self._check_str_is_prot(enzyme):
            enzyme = self._get_AAseq(enzyme, organism)

        return substrate, enzyme

    @overrides
    def _load_model(self, pretrained_state=None):
        try:
            self.model = pickle.load(open(path+"/KM_prediction_function/trained_model/"+pretrained_state, "rb"))

            #loading ESM-1b model:
            self._enzyme_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            self._batch_converter = alphabet.get_batch_converter()

        except:
            raise FileNotFoundError("No file named "+pretrained_state+" in "+path+"/KM_prediction_function/trained_model/")

    @overrides
    def estimate(self, substrates: list, enzymes:list, organisms=None, full_report = False):
        """
        Estimate kcat based on SMILES and protein sequence. If substrate is not SMILES, we will try to fetch if from PubChem. If enzymes is not a protein sequence
        we will try to fetch if from UniProt based on EC number and organism. 

        Parameters
        ----------
        substrates : list
            list of SMILES, or species names
        enzymes : list
            list of protein sequences, or EC numbers. If EC numbers, 'organims' must be specified, defaults to 'Escherichia coli'
        organisms : list, optional
            list of organisms names
        full_report : flag to return estimates, or also the process SMILES and protein sequences

        Returns
        -------
        list
            list of estimated predictions, if full_report is False
        dict
            dict of inputs, processed inputs, and estimates, if full_report is True
        """
                
        if len(substrates) != len(enzymes): raise 'Mismatch in the number of substrates and enzymes'
        if not organisms: organisms = ['Escherichia coli']*len(enzymes)
        processed_inputs = list(map(self._preprocess_inputs, substrates, enzymes, organisms))
        smiles = [i[0] for i in processed_inputs]
        seqs = [i[1] for i in processed_inputs]

        # get all the Nones out
        indexes = []
        for i,(S,s) in enumerate(zip(smiles, seqs)):
            if (S is not None) and (s is not None):
                indexes.append(i)

        #creating input matrices for all substrates:
        df_met = metabolite_preprocessing(metabolite_list = [smiles[i] for i in indexes])
        df_met = calculate_gnn_representations(df_met)
        #remove temporary metabolite directory:
        shutil.rmtree(path+"/KM_prediction_function/trained_model/temp_met")

        df_enzyme = calcualte_esm1b_vectors(self._enzyme_model, self._batch_converter, enzyme_list = [seqs[i] for i in indexes])

        fingerprints = np.array(list(df_met["GNN rep"]))
        ESM1b = np.array(list(df_enzyme["enzyme rep"]))

        print(len(fingerprints))
        print(len([smiles[i] for i in indexes]))

        print(len(ESM1b))
        print(len([seqs[i] for i in indexes]))

        X = np.concatenate([fingerprints, ESM1b], axis = 1)
        Kms = list(10**self.model.predict(xgb.DMatrix(X)))

        # put them back in order
        all_kms = [None]*len(enzymes)
        for k,i in enumerate(indexes):
            all_kms[i] = Kms[k]

        if full_report:
            return {'substrates':substrates,'SMILES':smiles, 'enzyme':enzymes,'seqs':seqs,'Km':all_kms}
        else: 
            return all_kms

ESTIMATORS = [ENKIE, DLKcat, KM_prediction]
