import unittest
from utils import *

class TestUtils(unittest.TestCase):

    def test_ice_espilon_relative(self):
        self.assertAlmostEqual(ice_epsilon_relative(273), 3.18849-1j*9.707E-04, 3)

    def test_ice_wave_number(self):
        self.assertAlmostEqual(ice_wave_number(425e6, 273), 15.9050860801277-0.00242114588766709j, 3)
        
    def test_uhf_radiation_efficiency(self):
        self.assertAlmostEqual(uhf_antenna.radiation_efficiency(273), 0.228, 2)

    def test_uhf_antenna_directivity(self):
        self.assertAlmostEqual(math_funcs.power_2_db(uhf_antenna.directivity(273)), 4.46, 3)
        
if __name__ == '__main__':
    unittest.main()