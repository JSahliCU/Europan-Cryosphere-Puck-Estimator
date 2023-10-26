import math_funcs
import numpy as np
from scipy.constants import e, epsilon_0, mu_0, k
from scipy.constants import c as c_0

class uhf_antenna():
    def directivity(T):
        directivity_in_100K_ice = 4.64 # dB
        directivity_in_273K_ice = 4.46 # dB
        m, b = math_funcs.linear_fit(
            100, math_funcs.db_2_power(directivity_in_100K_ice), 
            273, math_funcs.db_2_power(directivity_in_273K_ice))
        return m * T + b

    def radiation_efficiency(T):
        rad_eff_in_273K_ice = 0.228 # dB
        rad_eff_in_100K_ice = 0.223 # dB
        m, b = math_funcs.linear_fit(
            100, rad_eff_in_100K_ice, 
            273, rad_eff_in_273K_ice)
        return m * T + b
    
    match_efficiency = 0.952
    carrier_frequency = 413e6 # Hz

class hf_antenna():
    directivity = 2 # dB
    radiation_efficiency = 0.007
    match_efficiency = 1
    carrier_frequency = 5.373e6 # Hz

def ice_epsilon_relative(T):
    """
    Matzler and Wegmuller 1987 
    applicable 
        for pure ice
        for UHF/VHF bands 

    Args:
        T (float): temperature of the ice

    Returns:
        float: the relative dielectric constant of pure ice
    """
    T = T - 273.13 # convert from Kelvin to Celsius
    real = 3.1884 + 0.00091*T
    imag = 10**(-3.0129 + 0.0123*T)
    return real - 1j*imag

def ice_wave_number(frequency, T):
    """
    Shortcut function for calculating the wavenumber 
    using the ice_epsilon_relative function

    Args:
        frequency (float): frequency at which we are calculating the wave number
        T (float): temperature of the ice
    Returns:
        float: wave number
    """
    return 2 * np.pi * frequency * np.sqrt(epsilon_0 * mu_0) * np.sqrt(ice_epsilon_relative(T))

def ice_wavelength(frequency, T):
    return  np.pi * 2 / np.real(ice_wave_number(frequency, T))

def ice_loss_tangent(T):
    temp = ice_epsilon_relative(T)
    return np.imag(temp)/np.real(temp)