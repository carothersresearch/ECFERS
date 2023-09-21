import unittest
import numpy as np
from odbm.utils import *


"""Unit test for utils functions"""
def test_extractParams:
    param_str = 'k:100;k2:200'
    param_dict = extractParams(param_str)
    self.assertTrue(len(param_dict) == 2)
    self.assertTrue(param_dict['k'] == '100')

    bad_string = 'k:100,k2:200'
    self.assertRaises(NameError, extractParams(bad_string))

    bad_string = 'k = 100; k2 = 200'
    self.assertRaises(NameError, extractParams(bad_string))