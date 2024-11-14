# Imports and plotting setups
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math

import sys

import math_funcs

from scipy.special import loggamma, factorial, gamma
from decimal import Decimal
import decimal

from antenna_patterns import uhf_antenna, hf_antenna
from transmission_reflection_coefficients import transmissivity_fBm, reflectivity_fBm
from T_B_Europan_sky import angle_grid, T_B_sub_jovian_point, T_B_anti_jovian_point, T_B_anti_orbital_point

from scipy.constants import (
    epsilon_0,   # Permittivity of free space (vacuum)
    mu_0,        # Permeability of free space (vacuum)
    c,           # Speed of light in vacuum
    #e,           # Elementary charge
    #h,           # Planck constant
    #hbar,        # Reduced Planck constant (h-bar)
    k,           # Boltzmann constant
    #G,           # Newtonian constant of gravitation
    #m_e,         # Electron mass
    #m_p,         # Proton mass
    #m_n,         # Neutron mass
    #alpha,       # Fine-structure constant
    eV,          # Electron volt
)

def evaluate_number_of_pucks_on_arbitrary_europa(
        T_u = 104, #K
        T_l = 230, #K
        T_melt = 273.13, #K
        T_conv = 251.6, #K
        D_cond = 10.4e3, #m
        D_phi = 3.2e3, #m
        eta_vac = 0.1, #np.arrange(0, 0.3, 0.1)
        rho_salt = 1e-5, #np.linspace(1e-3, 4.2e-2, 10)
        D_conv = 5.8e3, #m
        delta_d = 10, #m
        H = 0.75, # Surface roughness hurst coefficient
        sigma_ref = 0.2, # Surface roughness at reference wavelength (at 1m)
        file_suffix = '',
        use_shannon_channel_limit=False,
        HF_communication_bandwidth=10e3,
        UHF_communication_bandwidth=10e3,
        HF_M_symbols=2,
        UHF_M_symbols=2,
        puck_RF_power=1.0, #W
        landing_site='sub jovian point'
    ):
    
    eim = europa_ice_model(
        T_u, 
        T_l, 
        T_melt, 
        T_conv, 
        D_cond, 
        D_phi, 
        eta_vac, 
        rho_salt, 
        D_conv, 
        delta_d,
    )

    mm = mission_model()
    uhf_ant = uhf_antenna()
    hf_ant = hf_antenna()
    ag = angle_grid(10, 10)

    # Evaluation of the sub-Jovian point on Europa
    # Rotate the brightness field pattern so we are standing on the sub-jovian point
    if landing_site == 'sub jovian point':
        brightness_field_of_europan_sky_vertical_low_band, \
            brightness_field_of_europan_sky_horizontal_low_band, \
                brightness_field_of_europan_sky_vertical_high_band, \
                    brightness_field_of_europan_sky_horizontal_high_band \
                        = T_B_sub_jovian_point()
    elif landing_site == 'anti jovian point':
        brightness_field_of_europan_sky_vertical_low_band, \
            brightness_field_of_europan_sky_horizontal_low_band, \
                brightness_field_of_europan_sky_vertical_high_band, \
                    brightness_field_of_europan_sky_horizontal_high_band \
                        = T_B_anti_jovian_point()
    elif landing_site == 'anti orbital point':
        brightness_field_of_europan_sky_vertical_low_band, \
            brightness_field_of_europan_sky_horizontal_low_band, \
                brightness_field_of_europan_sky_vertical_high_band, \
                    brightness_field_of_europan_sky_horizontal_high_band \
                        = T_B_anti_orbital_point()
    else:
        raise(ValueError('landing_site must be [\'sub jovian point\', \'anti jovian point\', \'anti orbital point\']'))

    # Integrate the transmitted field by summing to find transmitted field over all incident fields
    # Low band
    T_Bv_low_band, T_Bh_low_band = integrate_gamma_ab_and_T_Ba_i(
        ag, brightness_field_of_europan_sky_vertical_low_band, 
        brightness_field_of_europan_sky_horizontal_low_band,
        mm.low_band_f, 1, 
        eim.cryosphere_model_df['epsilon_s_prime'].values[0], 
        sigma_ref * (c/mm.low_band_f)**(H), H, 'transmission'
    )
    # High band
    T_Bv_high_band, T_Bh_high_band = integrate_gamma_ab_and_T_Ba_i(
        ag, brightness_field_of_europan_sky_vertical_high_band, 
        brightness_field_of_europan_sky_horizontal_high_band,
        mm.high_band_f, 1, 
        eim.cryosphere_model_df['epsilon_s_prime'].values[0], 
        sigma_ref * (c/mm.high_band_f)**(H), H, 'transmission'
    )

    T_Bv_low_band = T_Bv_low_band.reshape(ag.theta_grid.shape) 
    T_Bh_low_band = T_Bh_low_band.reshape(ag.theta_grid.shape) 
    T_Bv_high_band = T_Bv_high_band.reshape(ag.theta_grid.shape) 
    T_Bh_high_band = T_Bh_high_band.reshape(ag.theta_grid.shape) 

    # Calculate the downwelling noise stream
    eim.cryosphere_model_df['T_A Downwelling Low Band (K)'], \
        T_Bh_d_low_band, T_Bv_d_low_band = calc_welling_noise_stream(
        'down', mm.low_band_f, eim, ag, T_Bh_low_band, T_Bv_low_band, hf_ant.directivity_pattern_RHCP)
    eim.cryosphere_model_df['T_A Downwelling High Band (K)'], \
        T_Bh_d_high_band, T_Bv_d_high_band = calc_welling_noise_stream(
        'down', mm.high_band_f, eim, ag, T_Bh_high_band, T_Bv_high_band, uhf_ant.directivity_pattern_RHCP)

    # Estimate of the oceans emissivity using reflection
    epsilon_ocean_prime = 77 # Roger Lang
    epsilon_ocean_primeprime = 48

    theta_t_values = ag.theta_grid.flatten()
    phi_t_values = ag.phi_grid.flatten()

    theta_i_values = ag.theta_grid.flatten()
    phi_i_values = ag.phi_grid.flatten()

    n_i = np.sqrt(eim.cryosphere_model_df.iloc[-1]['epsilon_s_prime'] * epsilon_0 * mu_0)
    n_t = np.sqrt(epsilon_ocean_prime * epsilon_0 * mu_0)

    R_v = np.zeros_like(ag.theta_grid).flatten()
    R_h = np.zeros_like(ag.theta_grid).flatten()

    #Lambertian reflection 
    for t in np.arange(ag.theta_grid.size):
        theta_t = theta_t_values[t]
        phi_t = phi_t_values[t]

        for i in np.arange(ag.theta_grid.size):
            theta_i = theta_t_values[i]
            phi_i = phi_t_values[i]

            R_v[i] += np.cos(theta_t)*np.cos(theta_i)

            R_h[i] += np.cos(theta_t)*np.cos(theta_i)
            
    # # Specular reflection code in case you want to change over to that
    # for i in np.arange(ag.theta_grid.size):
    #     theta_i = theta_t_values[i]
    #     phi_i = phi_t_values[i]

    #     R_v[i] += (n_i * np.cos(theta_i) - n_t * np.sqrt(1 - (np.sin(theta_i) * n_i / n_t)**2)) \
    #         / (n_i * np.cos(theta_i) + n_t * np.sqrt(1 - (np.sin(theta_i) * n_i / n_t)**2))

    #     R_h[i] += (n_i * np.sqrt(1 - (n_i * np.sin(theta_i) / n_t)**2) - n_t * np.cos(theta_i)) \
    #         / (n_i * np.sqrt(1 - (n_i * np.sin(theta_i) / n_t)**2) + n_t * np.cos(theta_i))

    R_v = R_v.reshape(ag.theta_grid.shape)
    R_h = R_h.reshape(ag.theta_grid.shape)

    emissivity_ocean_v = (1 - R_v**2)
    emissivity_ocean_h = (1 - R_h**2)

    T_Bv_ocean_low_band = (emissivity_ocean_v*T_melt) + R_v**2 * T_Bv_d_low_band
    T_Bh_ocean_low_band = (emissivity_ocean_h*T_melt) + R_h**2 * T_Bh_d_low_band
    T_Bv_ocean_high_band = (emissivity_ocean_v*T_melt) + R_v**2 * T_Bv_d_high_band
    T_Bh_ocean_high_band = (emissivity_ocean_h*T_melt) + R_h**2 * T_Bh_d_high_band

    # Calculate the upwelling noise stream
    eim.cryosphere_model_df['T_A Upwelling Low Band (K)'], \
        T_Bh_d_low_band, T_Bv_d_low_band = calc_welling_noise_stream(
        'up', mm.low_band_f, eim, ag, 
        T_Bh_ocean_low_band , T_Bv_ocean_low_band ,
        hf_ant.directivity_pattern_RHCP)
    eim.cryosphere_model_df['T_A Upwelling High Band (K)'], \
        T_Bh_d_high_band, T_Bv_d_high_band = calc_welling_noise_stream(
        'up', mm.high_band_f, eim, ag, 
        T_Bh_ocean_high_band, T_Bv_ocean_high_band,
        uhf_ant.directivity_pattern_RHCP)

    # Calculate the downwelling noise stream including the upwelling noise stream reflection
    # Low band
    T_Bv_low_band_ref, T_Bh_low_band_ref = integrate_gamma_ab_and_T_Ba_i(
        ag, T_Bv_d_low_band, T_Bh_d_low_band,
        mm.low_band_f, eim.cryosphere_model_df['epsilon_s_prime'].values[0], 
        1 , 
        sigma_ref * (c/mm.low_band_f)**(H), H, 'reflection'
    )
    # High band
    T_Bv_high_band_ref, T_Bh_high_band_ref = integrate_gamma_ab_and_T_Ba_i(
        ag, T_Bv_d_high_band, T_Bh_d_high_band,
        mm.high_band_f, eim.cryosphere_model_df['epsilon_s_prime'].values[0], 
        1, 
        sigma_ref * (c/mm.high_band_f)**(H), H, 'reflection'
    )

    T_Bv_low_band_ref = T_Bv_low_band_ref.reshape(ag.theta_grid.shape) 
    T_Bh_low_band_ref = T_Bh_low_band_ref.reshape(ag.theta_grid.shape) 
    T_Bv_high_band_ref = T_Bv_high_band_ref.reshape(ag.theta_grid.shape) 
    T_Bh_high_band_ref = T_Bh_high_band_ref.reshape(ag.theta_grid.shape) 

    eim.cryosphere_model_df['T_A Downwelling Low Band (K)'], \
        T_Bh_d_low_band, T_Bv_d_low_band = calc_welling_noise_stream(
        'down', mm.low_band_f, eim, ag, 
        T_Bh_low_band + T_Bh_low_band_ref, 
        T_Bv_low_band + T_Bv_low_band_ref, 
        hf_ant.directivity_pattern_RHCP)
    eim.cryosphere_model_df['T_A Downwelling High Band (K)'], \
        T_Bh_d_high_band, T_Bv_d_high_band = calc_welling_noise_stream(
        'down', mm.high_band_f, eim, ag, 
        T_Bh_high_band + T_Bh_high_band_ref, 
        T_Bv_high_band + T_Bv_high_band_ref, 
        uhf_ant.directivity_pattern_RHCP)

    # ------- Now estimate the puck placements -----------
    # Estimate the puck placement at UHF
    uhf_pucks, uhf_pucks_cond, uhf_pucks_conv = estimate_puck_placement(
        puck_RF_power,
        UHF_communication_bandwidth,
        uhf_antenna(), #antenna_pattern type
        mm.high_band_f,
        1e3,
        1e-5,
        11,
        eim,
        'T_A Upwelling High Band (K)',
        'T_A Downwelling High Band (K)',
        UHF_M_symbols,
        True,
        file_suffix,
        use_shannon_channel_limit
    )

    hf_pucks, hf_pucks_cond, hf_pucks_conv = estimate_puck_placement(
        puck_RF_power,
        HF_communication_bandwidth,
        hf_antenna(), #antenna_pattern type
        mm.low_band_f,
        1e3,
        1e-5,
        11,
        eim,
        'T_A Upwelling Low Band (K)',
        'T_A Downwelling Low Band (K)',
        HF_M_symbols,
        True,
        file_suffix,
        use_shannon_channel_limit
    )
    return uhf_pucks, uhf_pucks_cond, uhf_pucks_conv, hf_pucks, hf_pucks_cond, hf_pucks_conv

def calc_path_loss(omega, epsilon_s_prime, epsilon_s_primeprime, sigma_s):
    return (omega / np.sqrt(2)) * np.sqrt(epsilon_0 * mu_0)\
        * np.sqrt(np.sqrt(epsilon_s_prime**2 + (epsilon_s_primeprime + sigma_s / (epsilon_0 * omega))**2) \
            - epsilon_s_prime)

def calc_gamma_ab(
    ag, f, epsilon_i_prime, epsilon_t_prime, 
    sigma, H, bistatic_polarization, torr):

    if torr == 'transmission':
        fBm_func = transmissivity_fBm
    elif torr == 'reflection':
        fBm_func = reflectivity_fBm
    else:
        raise ValueError('torr must be set to reflection or transmission')
    
    flat_shape = ag.theta_grid.flatten().shape[0]
    twoD_shape = ag.theta_grid.shape

    gamma_ab = np.zeros((flat_shape, twoD_shape[0]))
    theta_i_values = np.deg2rad(ag.theta.flatten())
    theta_t_values = np.deg2rad(ag.theta_grid.flatten())
    phi_t_values = np.deg2rad(ag.phi_grid.flatten())

    for t in np.arange(flat_shape):
        theta_t = theta_t_values[t]
        phi_t = theta_t_values[t]
        if theta_t < np.pi/2:
            for i in np.arange(twoD_shape[0]):
                theta_i = theta_i_values[i]
                if theta_i < np.pi/2:
                    gamma_ab[t, i] = np.nan_to_num(fBm_func(
                            f, epsilon_i_prime, epsilon_t_prime, sigma, H,
                            theta_i, np.pi - theta_t, phi_t, bistatic_polarization
                        ))
    return gamma_ab

def integrate_gamma_ab_and_T_Ba_i(
        ag, T_Bv_i, T_Bh_i,
        f, epsilon_i_prime, epsilon_t_prime, 
        sigma, H, torr):

    d_solid_angle = ag.d_solid_angle[None, :, :]

    gamma_ab_helper = lambda x: calc_gamma_ab(
        ag, f, epsilon_i_prime, epsilon_t_prime, 
        sigma, H, x, torr
    )
    # The gamma_ab function returns a function thats 
    # flattened along the transmitted angle grid
    # and has values in its 3rd (index =2) dimension  
    # representing incident theta. As gamma_ab is 
    # symmetric in phi, there is no 4th (index=3) dimension. 
    # To get the transmitted radiation (T_ba)
    # multiply the incident radiation (T_Bb) 
    # against the incident pattern theta pattern. 
    # To do this add a phi axis [:,:,None] to gamma_ab pattern
    # so the T_Bb pattern broadcasts along that axis (phi axis), 
    # then add an axis to the front of T_Bb [None, :, :] 
    # so we multiply incident radiation by incident gamma pattern.
    # The last two axis of expression 
    # 'gamma_ab_helper('vv')[:,:,None] * T_Bv[None,:,:]' 
    # represent the contribution of incident radiation 
    # to the transmitted angle described by the first axis.
    # By summing the last two axis using np.sum (---, (-1, -2)) 
    # the total contribution at every transmitted angle is found.
    T_Bv = gamma_ab_helper('vv')[:,:,None] * T_Bv_i[None,:,:] 
    T_Bv += gamma_ab_helper('vh')[:,:,None] * T_Bh_i[None,:,:]
    T_Bv = np.sum(T_Bv * d_solid_angle, (-1, -2)) / (4 * np.pi)

    T_Bh = gamma_ab_helper('hh')[:,:,None] * T_Bh_i[None,:,:] 
    T_Bh += gamma_ab_helper('hv')[:,:,None] * T_Bv_i[None,:,:]
    T_Bh = np.sum(T_Bh * d_solid_angle, (-1, -2)) / (4 * np.pi)

    return T_Bv, T_Bh

def calc_welling_noise_stream(
        direction, f, eim, ag, T_Bh, T_Bv, directivity_RHCP):
    
    if direction == 'down':
        depth_list = np.arange(len(eim.cryosphere_model_df))
    elif direction == 'up':
        depth_list = np.flip(np.arange(len(eim.cryosphere_model_df)))
    else:
        raise(ValueError('direction needs to be \'up\' or \'down\''))

    T_A = np.zeros(len(depth_list))

    T_Bh_d = T_Bh.copy()
    T_Bv_d = T_Bv.copy()

    delta_path_loss = np.ones(ag.theta.shape)

    exp_numerator = np.ones(ag.theta.shape)
    for d in depth_list:
        epsilon_s_prime = eim.cryosphere_model_df.loc[d]['epsilon_s_prime']
        epsilon_s_primeprime = eim.cryosphere_model_df.loc[d]['epsilon_s_primeprime']
        sigma_s = eim.cryosphere_model_df.loc[d]['sigma_s']
        temperature_ice = eim.cryosphere_model_df.loc[d]['Temperature (K)']
        alpha = calc_path_loss(f, epsilon_s_prime, epsilon_s_primeprime, sigma_s)
        
        # The loss is symmetric in phi by homogenous layer assumption
        for t in np.arange(len(ag.theta)):
            theta_t = np.deg2rad(ag.theta[t])
            if theta_t < np.pi/2:
                exp_numerator[t] = -2 * (eim.delta_d / np.abs(np.cos(theta_t))) * alpha    
                delta_path_loss[t] = 1 - np.e**(exp_numerator[t])
            else: # if we are looking down, set the TA contribution of this depth to 0
                exp_numerator[t] = 0
                delta_path_loss[t] = 0
        
        # Accumulate ice contribution and loss
        if d > 0:
            T_Bh_d += \
                -1 * delta_path_loss[:, None] * T_Bh_d \
                    - exp_numerator[:, None] * temperature_ice
            T_Bv_d += \
                -1 * delta_path_loss[:, None] * T_Bv_d \
                    - exp_numerator[:, None] * temperature_ice

        # Integrate over antenna directivity to get antenna temperature
        temp = (T_Bh_d \
            + T_Bv_d).reshape(ag.theta_grid.shape) * directivity_RHCP \
            * np.sin(np.deg2rad(ag.theta))[:, None]
        temp = scipy.integrate.simpson(
            temp, 
            x=np.deg2rad(ag.phi_grid))
        T_A[d] = (1 / (4 * np.pi)) * scipy.integrate.simpson(
            temp, 
            x=np.deg2rad(ag.theta))
        
    return T_A, T_Bh_d, T_Bv_d

class mission_model:
    def __init__(self,
        low_band_f = 5.373e6,
        high_band_f = 413e6,
        max_BER = 10**-5,
        bit_rate = 1e3, # bps
        link_BW = 10e3, # 3.43e3 # Hz
        P_t = 1 # W
        ):
        
        self.max_BER = max_BER
        self.bit_rate = bit_rate
        self.link_BW = link_BW
        self.P_t = P_t
        
        self.low_band_f = low_band_f
        self.low_band_omega = 2 * np.pi * low_band_f
        self.high_band_f = high_band_f
        self.high_band_omega = 2 * np.pi * high_band_f

class europa_ice_model:
    def __init__(self,
        T_u = 104, #K
        T_l = 230, #K
        T_melt = 273.13, #K
        T_conv = 251.6, #K
        D_cond = 10.4e3, #m
        D_phi = 3.2e3, #m
        eta_vac = 0.1, #np.arrange(0, 0.3, 0.1)
        rho_salt = 1e-5, #np.linspace(1e-3, 4.2e-2, 10)
        D_conv = 5.8e3, #m
        delta_d = 10 #m
        ):

        self.T_u = T_u 
        self.T_l = T_l 
        self.T_melt = T_melt 
        self.T_conv = T_conv 
        self.D_cond = D_cond 
        self.D_phi = D_phi 
        self.eta_vac = eta_vac 
        self.rho_salt = rho_salt 
        self.D_conv = D_conv 
        self.delta_d = delta_d
        
        self.D_total = self.D_cond + self.D_conv

        self.cryosphere_model_df = pd.DataFrame({'Depth (m)':\
            np.arange(0, self.D_total+delta_d, delta_d)})
        self.cryosphere_model_df['Temperature (K)'] = \
            self.cryosphere_model_df['Depth (m)'].map(self.temperature_at_depth)
        self.cryosphere_model_df['Porosity (m^3/m^3)'] = \
            self.cryosphere_model_df['Depth (m)'].map(self.porosity_at_depth)
        self.cryosphere_model_df['Salt fraction (kg/kg)'] = \
            self.cryosphere_model_df['Depth (m)'].map(self.salt_fraction_at_depth)

        # Estimate the dielectric constant of the ice
        T = self.cryosphere_model_df['Temperature (K)']

        self.cryosphere_model_df['epsilon_s_prime'] = 3.1884 + 0.00091*(T - 273.13)
        self.cryosphere_model_df['epsilon_s_primeprime'] = 10**(-3.0129 + 0.0123*(T - 273.13))

        # Modify the epsilon s primes by the Maxwell-Garnett
        epsilon_s_prime = self.cryosphere_model_df['epsilon_s_prime']
        epsilon_s_primeprime = self.cryosphere_model_df['epsilon_s_primeprime']
        eta_vac = self.cryosphere_model_df['Porosity (m^3/m^3)']
        epsilon_s = (epsilon_s_prime - 1j * epsilon_s_primeprime) * epsilon_0
        epsilon_m = epsilon_s + (3 * eta_vac * epsilon_s * (epsilon_0 - epsilon_s)) \
            / ((2 * epsilon_s + epsilon_0) - eta_vac * (epsilon_0 - epsilon_s))

        self.cryosphere_model_df['epsilon_s_prime'] = np.real(epsilon_m / epsilon_0)
        self.cryosphere_model_df['epsilon_s_primeprime'] = -1 * np.imag(epsilon_m / epsilon_0)

        # Estimate the conductivity of the ice
        molar_mass_salt = (35.453 + 22.990) / 1000 # kg/mol
        molarity_salt = self.cryosphere_model_df['Salt fraction (kg/kg)'] / molar_mass_salt
        micro_molarity_salt = molarity_salt * 1e6

        self.cryosphere_model_df['sigma_s'] = 1e-6 * \
            (9 * np.e ** ((0.58*eV/k) * (1 / 258.15 - 1 / T)) \
            + 0.55 * micro_molarity_salt * np.e ** ((0.22*eV/k) * (1 / 258.15 - 1 / T)))

    def temperature_at_depth(self, d):
        if d > self.D_cond + self.D_conv:
            return self.T_melt
        elif d > self.D_cond:
            return self.T_conv
        else:
            m, b = math_funcs.linear_fit(
                0, self.T_u, 
                self.D_cond, self.T_l)
            return m * d + b

    def porosity_at_depth(self, d):
        if d <= self.D_phi:
            return self.eta_vac
        else:
            return 0
        
    def salt_fraction_at_depth(self, d):
        return self.rho_salt

def estimate_puck_placement(
    transmitter_power,
    communication_bandwidth,
    antenna_pattern, #antenna_pattern type
    communication_frequency,
    data_rate,
    limit_probability_of_error,
    noise_figure,
    cryosphere_model,
    T_A_upwelling_col,
    T_A_downwelling_col,
    symbols,
    record_puck_placement_data,
    file_suffix,
    use_shannon_channel_limit,
):
    eim = cryosphere_model
    receiver_temperature = 290 * (math_funcs.db_2_power(noise_figure) - 1)
    attenuation = 1
    prop_distance = 0

    # Data to record
    placement_depths = []
    attenuations = []

    number_of_pucks = 0
    number_of_cond_pucks = 0
    number_of_conv_pucks = 0

    for d in np.arange(len(eim.cryosphere_model_df)):
        epsilon_s_prime = eim.cryosphere_model_df.loc[d]['epsilon_s_prime']
        epsilon_s_primeprime = eim.cryosphere_model_df.loc[d]['epsilon_s_primeprime']
        sigma_s = eim.cryosphere_model_df.loc[d]['sigma_s']
        temperature_ice = eim.cryosphere_model_df.loc[d]['Temperature (K)']
        
        lambda_s = c / (np.sqrt(epsilon_s_prime) * communication_frequency)

        lower_gain = antenna_pattern.realized_gain(temperature_ice)
        lower_rad_eff = antenna_pattern.radiation_efficiency(temperature_ice)
        lower_match_eff = antenna_pattern.matching_efficiency(temperature_ice)
        
        lower_T_A_upwelling = eim.cryosphere_model_df.loc[d][T_A_upwelling_col]
        lower_T_A_downwelling = eim.cryosphere_model_df.loc[d][T_A_upwelling_col]

        ant_backside_exists = \
            antenna_pattern.directivity(180, 0, temperature_ice) > 0
        
        alpha = calc_path_loss(
            2 * np.pi * communication_frequency, epsilon_s_prime, 
            epsilon_s_primeprime, sigma_s)
        differential_attenuation = np.e**(-2 * (eim.delta_d) * alpha)
        attenuation *= differential_attenuation

        prop_distance += eim.delta_d

        def calc_prob_error(
                lambda_s_receiver,
                lambda_s_transmitter,
                T_A_upwelling,
                T_A_downwelling,
                rad_eff,
                match_eff,
                temperature_ice
        ):
            received_power = transmitter_power * upper_gain * lower_gain \
                *  (lambda_s_receiver**2 / (4 * np.pi * prop_distance)**2)\
                * attenuation * (lambda_s_transmitter / lambda_s_receiver)
            
            ant_temp = rad_eff * match_eff * T_A_upwelling\
                + rad_eff * match_eff * T_A_downwelling\
                + (1 - rad_eff) * match_eff * temperature_ice \
                + (1 - match_eff) * 290
            sys_temp = ant_temp + receiver_temperature
            noise_power = k * sys_temp * communication_bandwidth 

            signal_to_noise_ratio = received_power / noise_power

            if use_shannon_channel_limit:
                if 2**(data_rate/communication_bandwidth) - 1 < signal_to_noise_ratio:
                    probability_of_error = 0
                else:
                    probability_of_error = 1
            else:
                CNR_per_bit = (data_rate/communication_bandwidth) * signal_to_noise_ratio
                probability_of_error = probability_of_error_for_MFSK(CNR_per_bit, symbols)

            return probability_of_error
        
        place_puck = False
        if d == 0 or d==len(eim.cryosphere_model_df)-1:
            place_puck = True
        else:
            lower_lambda_s = lambda_s
            # Estimate SNR at upper puck
            upper_prob_error = calc_prob_error(
                upper_lambda_s,
                lower_lambda_s,
                upper_T_A_upwelling,
                upper_T_A_downwelling * ant_backside_exists,
                upper_rad_eff,
                upper_match_eff,
                upper_ice_temp
            )
            # Estimate SNR at lower puck
            lower_prob_error = calc_prob_error(
                lower_lambda_s,
                upper_lambda_s,
                lower_T_A_upwelling * ant_backside_exists,
                lower_T_A_downwelling,
                lower_rad_eff,
                lower_match_eff,
                temperature_ice
            )

            if lower_prob_error > limit_probability_of_error \
                or upper_prob_error > limit_probability_of_error:
                place_puck = True

        if place_puck:
            number_of_pucks += 1
            if d * eim.delta_d > eim.D_cond:
                number_of_conv_pucks += 1
            else:
                number_of_cond_pucks += 1

            # Reset accumulating variables
            attenuation = 1
            prop_distance = 0

            # Set upper puck static values to current lower puck values
            upper_lambda_s = lambda_s
            upper_ice_temp = temperature_ice
            upper_gain = lower_gain
            upper_rad_eff = lower_rad_eff
            upper_match_eff = lower_match_eff
            upper_T_A_upwelling = lower_T_A_upwelling
            upper_T_A_downwelling = lower_T_A_downwelling

            # Record Data
            placement_depths.append(eim.cryosphere_model_df.loc[d]['Depth (m)'])
            attenuations.append(attenuation)

    if record_puck_placement_data:
        recorded_data = pd.DataFrame({
            'Placement Depth (m)': placement_depths, 
            'Attenuation': attenuations})
        
        recorded_data.to_csv('recorded_data_low_band' + file_suffix + '.csv')
        eim.cryosphere_model_df.to_csv('cryosphere_model' + file_suffix + '.csv')

    return number_of_pucks, number_of_cond_pucks, number_of_conv_pucks

# Bit error rate for MFSK with N bits
def probability_of_error_for_MFSK(CNR_per_bit, N):
    if 10*np.log10(CNR_per_bit) > 15:
        return 10**-40
    elif CNR_per_bit > 4*np.log(2):
        return np.e**(-(N/2) * (CNR_per_bit - 2 * np.log(2)))
    else:
        return np.e**(-1 * N * ((np.sqrt(CNR_per_bit) - np.sqrt(np.log(2)))**2))

if __name__ == "__main__":
    import time
    start_time = time.time()
    uhf_pucks, uhf_pucks_cond, uhf_pucks_conv, hf_pucks, hf_pucks_cond, hf_pucks_conv =\
        evaluate_number_of_pucks_on_arbitrary_europa()
    print(f'HF Pucks {hf_pucks} and UHF pucks {uhf_pucks}')
    print("--- %s seconds ---" % (time.time() - start_time))
    