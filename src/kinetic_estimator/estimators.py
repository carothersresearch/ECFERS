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
warnings.filterwarnings("ignore")

path = os.getcwd()+'/src/kinetic_estimator'

class BaseEstimator:
    """
    
    
    """
    name = 'DLKcat'
    parameter = ['kcat']
    inputs = ['species', 'enzyme sequence']

    def _load_model(self):
        """
        Writes rate of chemical reaction. This function should always be overriden.

        Returns
        -------
        str
        """
        pass

        # One method to obtain SMILES by PubChem API using the website
    def _get_smiles(self, name):
        try :
            url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT' % name
            req = requests.get(url)
            if req.status_code != 200:
                smiles = None
            else:
                smiles = req.content.splitlines()[0].decode()
        except:
            raise LookupError("Could not get SMILES string for "+name+"from PubChem. Please provide SMILES string instead of species name manually.")
        return smiles
    
    def _get_AAseq(self, uniprot):
        AAseq = None
        if uniprot is str:
            uniprot_url = 'https://rest.uniprot.org/uniprotkb/'+uniprot+'.json'
            try:
                response = requests.get(uniprot_url)
                # Check if the request was successful (status code 200)
                if response.status_code == 200:
                    # Parse the JSON data from the response
                    data = json.loads(response.text)
                    AAseq = data['sequence']['value']
                else:
                    print(f"Failed to retrieve data. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
        return AAseq

    def estimate(self, substrates:list, enzymes:list) -> list:
        """
        Writes rate of chemical reaction. This function should always be overriden.

        Returns
        -------
        str
        """
        pass

class Estimator:
    """
    _summary_

    Parameters
    ----------
    estimator : str
        _description_
    parameter : str
        _description_
    """
    def __init__(self, estimator:str, parameter:str):
        self.estimator = None
        self.estimator_dict = {}
        [self.addEstimator(e) for e in ESTIMATORS]
        if estimator in [e.name for e in ESTIMATORS]:
            if parameter in self.estimator_dict[estimator].parameter:
                self.estimator = self.estimator_dict[estimator]()
            else:
                raise KeyError("Estimator " + estimator + " cannot be used to estimate " + parameter)
        else:
            raise KeyError("No estimator named " + estimator + " found. Available estimators are: " + [e.name for e in ESTIMATORS])
        
    def addEstimator(self, new_estimator:BaseEstimator):
        """Adds a new Mechanism to the internal mechanism dictionary 

        Parameters
        ----------
        new_mechanism (Mechanism): Mechanism class 
        """
        self.estimator_dict[new_estimator.name] = new_estimator
    
    def estimate(self, substrates:list, enzymes:list) -> float:
        return self.estimator.estimate(substrates, enzymes)

class ENKIE(BaseEstimator):
    name = 'ENKIE'
    parameter = ['kcat','Km']
    """
    
    
    """
    def __init__(self) -> None:
        pass

class DLKcat(BaseEstimator):
    name = 'DLKcat'
    parameter = ['kcat']
    inputs = ['species', 'enzyme sequence']

    def __init__(self, pretrained_state = 'default_pretrained_model') -> None:
        super().__init__()
        self._load_model(pretrained_state)

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
    def estimate(self, substrates: list, enzymes:list) -> list:
        kcats = []
        for s, e in zip(substrates, enzymes):
            try:
                smiles = self._get_smiles(s)
            except Exception as ex:
                raise ex
            
            try:
                seq = self._get_AAseq(e)
            except Exception as ex:
                raise ex
        
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
    name = 'KM_prediction'
    parameter = ['Km']
    inputs = ['species', 'enzyme sequence'] # or uniprot, ideally handle both 

    def __init__(self, pretrained_state = 'xgboost_model_new_KM_esm1b.dat') -> None:
        super().__init__()
        self._load_model(pretrained_state)

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
    def estimate(self, substrates: list, enzymes:list ) -> list:
        smiles = list(map(self._get_smiles, substrates))
        seqs = list(map(self._get_AAseq, enzymes))

        #creating input matrices for all substrates:
        df_met = metabolite_preprocessing(metabolite_list = smiles)
        df_met = calculate_gnn_representations(df_met)
        #remove temporary metabolite directory:
        shutil.rmtree(path+"/KM_prediction_function/trained_model/temp_met")

        df_enzyme = calcualte_esm1b_vectors(self._enzyme_model, self._batch_converter, enzyme_list = seqs)

        fingerprints = np.array(list(df_met["GNN rep"]))
        ESM1b = np.array(list(df_enzyme["enzyme rep"]))
        X = np.concatenate([fingerprints, ESM1b], axis = 1)
        dX = xgb.DMatrix(X)
        
        return list(10**self.model.predict(dX))

ESTIMATORS = [ENKIE, DLKcat, KM_prediction]
