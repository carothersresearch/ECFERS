import enkie
import src.kinetic_estimator.DLKcat.model as DLKcat_model
import torch
from overrides import EnforceOverrides, overrides, final
import requests
from rdkit import Chem
import numpy as np
from collections import defaultdict
import os

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
    
    def estimate(self, inputs : list) -> float:
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
    
    def estimate(self, inputs:list) -> float:
        return self.estimator.estimate(inputs)

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
            self.model.load_state_dict(torch.load(path+'/DLKcat/trained_model/'+pretrained_state, map_location=self._device))

    @overrides
    def estimate(self, inputs: list) -> float:
        try:
            smiles = self._get_smiles(inputs[0])
        except:
            raise LookupError("Could not get SMILES string for "+input[0]+"from PubChem. Please provide SMILES string instead of species name manually.")
       
        radius=2
        ngram=3
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = self._create_atoms(mol)
        i_jbond_dict = self._create_ijbonddict(mol)
        fingerprints = self._extract_fingerprints(atoms, i_jbond_dict, radius)
        adjacency = self._create_adjacency(mol)
        words = self._split_sequence(inputs[1],ngram)

        fingerprints = torch.LongTensor(fingerprints).to(self._device)
        adjacency = torch.FloatTensor(adjacency).to(self._device)
        words = torch.LongTensor(words).to(self._device)

        return 2**self.model.forward([fingerprints, adjacency, words]).item()
    
    # One method to obtain SMILES by PubChem API using the website
    def _get_smiles(self, name):
        try :
            url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT' % name
            req = requests.get(url)
            if req.status_code != 200:
                smiles = None
            else:
                smiles = req.content.splitlines()[0].decode()

        except :
            smiles = None
        return smiles
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

ESTIMATORS = [ENKIE, DLKcat]
