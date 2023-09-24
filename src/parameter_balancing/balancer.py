
from symbolicSBML import SBMLModel, Parameters
from parameter_balancer import Balancer, ParameterData
import numpy as np

# First test case with semi realistic values.
# Priors are linear scale here and will be converted inside.
priors_r = {
    # Base quantities
    Parameters.mu: (-880.0, 680.00),
    Parameters.kv: (10.0, 6.26),
    Parameters.km: (0.1, 6.26),
    Parameters.c: (0.1, 10.32),
    Parameters.u: (0.0001, 10.32),
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

metabolites = ['G6P', 'F6P']
reactions = ['PGI']
data = [
    (Parameters.c, 'G6P', None, 10.0, 1.0),
    (Parameters.c, 'F6P', None, 10.0, 1.0),
    (Parameters.km, 'G6P', 'PGI', 0.28, 0.056),
    (Parameters.km, 'F6P', 'PGI', 0.147, 0.0294),
    (Parameters.keq, None, 'PGI', 0.361, 0.0361),
    (Parameters.vmax, None, 'PGI', 1511, 151),
]
S = np.zeros((len(metabolites), len(reactions)))
S[:, 0] = [-1, 1]

data_r = ParameterData(*zip(*data))
structure_r = SBMLModel.from_structure(metabolites, reactions, S)
test = Balancer(priors_r, data_r, structure_r, T=300, R=8.314 / 1000., augment=True).balance(sparse=False)

print(test.to_frame().T)
