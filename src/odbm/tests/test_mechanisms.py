"""Unit tests for Mechanisms module"""
import unittest
import numpy as np
from odbm.mechanisms import *

# Does Mechanism class contain reactions or mechanism subclasses?
# How to test writeRate for each mechanism?

class Test_Mechanism(unittest.TestCase):
    """Unit test for Mechanism parent class"""
    def test___init___(self):
        #Test proper initialization of class with proper inputs
        rxn = {'Enzyme':'',
        'Substrate':'A',
        'Product':'B',
        'Cofactor':'',
        'Parameters':'k:100',
        'Label':'R1'}

        self.mech = Mechanism(rxn)

        # check to make sure necessary attributes are inputted
        self.assertTrue(self.mech.enzyme is None)
        self.assertTrue(self.mech.substrates is not None)
        self.assertFalse(self.mech.products is not None)
        self.assertTrue(self.mech.params is not None)
        self.assertTrue(self.mech.label is not None)

    def test_writeEquation(self):
        eq_str = self.mech.writeEquation()
        self.assertTrue(eq_str == 'R1 : A -> B')

