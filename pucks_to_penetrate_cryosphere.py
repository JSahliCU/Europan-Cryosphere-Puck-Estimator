# Imports and plotting setups
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math

import math_funcs

from scipy.special import loggamma, factorial, gamma
from decimal import Decimal
import decimal

from antenna_patterns import antenna
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
        file_suffix = ''
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
    ant = antenna()

    ag = angle_grid(10, 10)

    # Evaluation of the sub-Jovian point on Europa
    # Rotate the brightness field pattern so we are standing on the sub-jovian point
    brightness_field_of_europan_sky_vertical_low_band, \
        brightness_field_of_europan_sky_horizontal_low_band, \
            brightness_field_of_europan_sky_vertical_high_band, \
                brightness_field_of_europan_sky_horizontal_high_band \
                    = T_B_sub_jovian_point()

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
        'down', mm.low_band_f, eim, ag, T_Bh_low_band, T_Bv_low_band, ant.HF_directivity_pattern_RHCP)
    eim.cryosphere_model_df['T_A Downwelling High Band (K)'], \
        T_Bh_d_high_band, T_Bv_d_high_band = calc_welling_noise_stream(
        'down', mm.high_band_f, eim, ag, T_Bh_high_band, T_Bv_high_band, ant.UHF_directivity_pattern_RHCP)

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
        ant.HF_directivity_pattern_RHCP)
    eim.cryosphere_model_df['T_A Upwelling High Band (K)'], \
        T_Bh_d_high_band, T_Bv_d_high_band = calc_welling_noise_stream(
        'up', mm.high_band_f, eim, ag, 
        T_Bh_ocean_high_band, T_Bv_ocean_high_band,
        ant.UHF_directivity_pattern_RHCP)

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
        ant.HF_directivity_pattern_RHCP)
    eim.cryosphere_model_df['T_A Downwelling High Band (K)'], \
        T_Bh_d_high_band, T_Bv_d_high_band = calc_welling_noise_stream(
        'down', mm.high_band_f, eim, ag, 
        T_Bh_high_band + T_Bh_high_band_ref, 
        T_Bv_high_band + T_Bv_high_band_ref, 
        ant.UHF_directivity_pattern_RHCP)

    # ------- Now estimate the puck placements -----------
    high_band_antenna = uhf_antenna()
    low_band_antenna = hf_antenna()

    # while placement_depth < eim.D_total:
    bandwidth = 10e3 #Hz
    bit_rate = 1e3 #bps
    limit_probability_of_error = 10**-5
    noise_figure = math_funcs.db_2_power(11)
    receiver_temperature = 290 * (noise_figure - 1)

    epsilon_s_prime = eim.cryosphere_model_df.loc[0]['epsilon_s_prime']
    epsilon_s_primeprime = eim.cryosphere_model_df.loc[0]['epsilon_s_primeprime']
    sigma_s = eim.cryosphere_model_df.loc[0]['sigma_s']
    temperature_ice = eim.cryosphere_model_df.loc[0]['Temperature (K)']

    transmitted_power = 1
    lambda_s_t = c / (np.sqrt(epsilon_s_prime) * mm.high_band_f)
    transmitter_gain = high_band_antenna.realized_gain(temperature_ice)

    radiation_efficiency = high_band_antenna.radiation_efficiency(temperature_ice)
    matching_efficiency = high_band_antenna.matching_efficiency
    T_A = eim.cryosphere_model_df.loc[0]['T_A Upwelling High Band (K)']

    antenna_temperature = radiation_efficiency * matching_efficiency * T_A \
        + (1 - radiation_efficiency) * matching_efficiency * temperature_ice \
            + (1 - matching_efficiency) * 290
    system_temperature = antenna_temperature + receiver_temperature
    noise_power_upper_puck = k * system_temperature * bandwidth

    meter_distance_per_wave = 0
    attenuation = 1
    uhf_number_of_pucks = 1

    # Data to record
    placement_depths = [0]
    received_powers = [0]
    noise_powers = [noise_power_upper_puck]
    attenuations = [attenuation]
    meter_distance_per_waves = [meter_distance_per_wave]
        
    for d in np.arange(len(eim.cryosphere_model_df)):
        epsilon_s_prime = eim.cryosphere_model_df.loc[d]['epsilon_s_prime']
        epsilon_s_primeprime = eim.cryosphere_model_df.loc[d]['epsilon_s_primeprime']
        sigma_s = eim.cryosphere_model_df.loc[d]['sigma_s']
        temperature_ice = eim.cryosphere_model_df.loc[d]['Temperature (K)']
        
        lambda_s = c / (np.sqrt(epsilon_s_prime) * mm.high_band_f)
        lambda_s_r = lambda_s

        alpha = calc_path_loss(mm.high_band_omega, epsilon_s_prime, epsilon_s_primeprime, sigma_s)
        
        differential_attenuation = np.e**(-2 * (eim.delta_d) * alpha)
        attenuation *= differential_attenuation
        
        differential_meter_distance_per_wave = lambda_s * eim.delta_d
        meter_distance_per_wave += differential_meter_distance_per_wave
        space_path_loss = 1 / (meter_distance_per_wave**2)
        
        # Received power at the lower puck and upper puck
        received_power = transmitted_power * (lambda_s_t**2/(4 * np.pi)) \
            * transmitter_gain \
                * attenuation * space_path_loss \
                    * (lambda_s_r**2/(4 * np.pi)) \
                        * high_band_antenna.realized_gain(temperature_ice)
        
        # Noise power at the lower puck
        radiation_efficiency = high_band_antenna.radiation_efficiency(temperature_ice)
        matching_efficiency = high_band_antenna.matching_efficiency
        T_A = eim.cryosphere_model_df.loc[d]['T_A Downwelling High Band (K)']
        
        antenna_temperature = radiation_efficiency * matching_efficiency * T_A \
            + (1 - radiation_efficiency) * matching_efficiency * temperature_ice \
                + (1 - matching_efficiency) * 290
        system_temperature = antenna_temperature + receiver_temperature
        noise_power = k * system_temperature * bandwidth

        # Calculate the probability of a bit error at the lower puck
        SNR = received_power / noise_power
        CNR_per_bit = (bit_rate/bandwidth) * SNR
        probability_of_error = probability_of_error_for_MFSK(CNR_per_bit, 2)

        # Calculate the probability of a bit error at the upper puck
        SNR_upper_puck = received_power / noise_power_upper_puck
        CNR_per_bit_upper_puck = (bit_rate/bandwidth) * SNR_upper_puck
        probability_of_error_upper_puck = probability_of_error_for_MFSK(CNR_per_bit_upper_puck, 2)

        # If we are going to pass the limit of error probability, place a puck
        # and then reset all of the upper puck stuff
        if probability_of_error > limit_probability_of_error \
            and probability_of_error_upper_puck > limit_probability_of_error:
            uhf_number_of_pucks += 1

            # Record data
            placement_depths.append(eim.cryosphere_model_df.loc[d]['Depth (m)'])
            received_powers.append(received_power)
            noise_powers.append(noise_power)
            attenuations.append(attenuation)
            meter_distance_per_waves.append(meter_distance_per_wave)

            # Reset upper puck things

            lambda_s_t = c / (np.sqrt(epsilon_s_prime) * mm.high_band_f)
            transmitter_gain = high_band_antenna.realized_gain(temperature_ice)

            radiation_efficiency = high_band_antenna.radiation_efficiency(temperature_ice)
            matching_efficiency = high_band_antenna.matching_efficiency
            T_A = eim.cryosphere_model_df.loc[d]['T_A Upwelling High Band (K)']

            antenna_temperature = radiation_efficiency * matching_efficiency * T_A \
                + (1 - radiation_efficiency) * matching_efficiency * temperature_ice \
                    + (1 - matching_efficiency) * 290
            system_temperature = antenna_temperature + receiver_temperature
            noise_power_upper_puck = k * system_temperature * bandwidth

            meter_distance_per_wave = 0
            attenuation = 1

    recorded_data = pd.DataFrame({
        'Placement Depth (m)': placement_depths, 
        'Received Power (W)': received_power, 
        'Noise Power (W)': noise_power,
        'Attenuation': attenuations,
        'meter_distance_per_waves': meter_distance_per_waves})
    recorded_data.to_csv('recorded_data_high_band' + file_suffix + '.csv')
    eim.cryosphere_model_df.to_csv('cryosphere_model' + file_suffix + '.csv')

    # Repeat the calculation for the lower band
    bandwidth = 10e3 #Hz
    bit_rate = 1e3 #bps
    limit_probability_of_error = 10**-5
    noise_figure = math_funcs.db_2_power(11)
    receiver_temperature = 290 * (noise_figure - 1)

    epsilon_s_prime = eim.cryosphere_model_df.loc[0]['epsilon_s_prime']
    epsilon_s_primeprime = eim.cryosphere_model_df.loc[0]['epsilon_s_primeprime']
    sigma_s = eim.cryosphere_model_df.loc[0]['sigma_s']
    temperature_ice = eim.cryosphere_model_df.loc[0]['Temperature (K)']

    transmitted_power = 1
    lambda_s_t = c / (np.sqrt(epsilon_s_prime) * mm.low_band_f)
    transmitter_gain = low_band_antenna.directivity * low_band_antenna.matching_efficiency * low_band_antenna.radiation_efficiency

    radiation_efficiency = low_band_antenna.radiation_efficiency
    matching_efficiency = low_band_antenna.matching_efficiency
    T_A = eim.cryosphere_model_df.loc[0]['T_A Downwelling Low Band (K)']

    antenna_temperature = radiation_efficiency * matching_efficiency * T_A \
        + (1 - radiation_efficiency) * matching_efficiency * temperature_ice \
            + (1 - matching_efficiency) * 290
    system_temperature = antenna_temperature + receiver_temperature
    noise_power_upper_puck = k * system_temperature * bandwidth

    meter_distance_per_wave = 0
    attenuation = 1
    hf_number_of_pucks = 1

    # Data to record
    placement_depths = [0]
    received_powers = [0]
    noise_powers = [noise_power_upper_puck]
    attenuations = [attenuation]
    meter_distance_per_waves = [meter_distance_per_wave]

    for d in np.arange(len(eim.cryosphere_model_df)):
        epsilon_s_prime = eim.cryosphere_model_df.loc[d]['epsilon_s_prime']
        epsilon_s_primeprime = eim.cryosphere_model_df.loc[d]['epsilon_s_primeprime']
        sigma_s = eim.cryosphere_model_df.loc[d]['sigma_s']
        temperature_ice = eim.cryosphere_model_df.loc[d]['Temperature (K)']
        
        lambda_s = c / (np.sqrt(epsilon_s_prime) * mm.low_band_f)
        lambda_s_r = lambda_s

        alpha = calc_path_loss(mm.low_band_omega, epsilon_s_prime, epsilon_s_primeprime, sigma_s)
        
        differential_attenuation = np.e**(-2 * (eim.delta_d) * alpha)
        attenuation *= differential_attenuation
        
        differential_meter_distance_per_wave = lambda_s * eim.delta_d
        meter_distance_per_wave += differential_meter_distance_per_wave
        space_path_loss = 1 / (meter_distance_per_wave**2)
        
        # Received power at the lower puck and upper puck
        received_power = transmitted_power * (lambda_s_t**2/(4 * np.pi)) \
            * transmitter_gain \
                * attenuation * space_path_loss \
                    * (lambda_s_r**2/(4 * np.pi)) \
                        * transmitter_gain
        
        # Noise power at the lower puck
        T_A = eim.cryosphere_model_df.loc[d]['T_A Downwelling Low Band (K)']
        
        antenna_temperature = radiation_efficiency * matching_efficiency * T_A \
            + (1 - radiation_efficiency) * matching_efficiency * temperature_ice \
                + (1 - matching_efficiency) * 290
        system_temperature = antenna_temperature + receiver_temperature
        noise_power = k * system_temperature * bandwidth

        # Calculate the probability of a bit error at the lower puck
        SNR = received_power / noise_power
        CNR_per_bit = (bit_rate/bandwidth) * SNR
        probability_of_error = probability_of_error_for_MFSK(CNR_per_bit, 2)

        # Calculate the probability of a bit error at the upper puck
        SNR_upper_puck = received_power / noise_power_upper_puck
        CNR_per_bit_upper_puck = (bit_rate/bandwidth) * SNR_upper_puck
        probability_of_error_upper_puck = probability_of_error_for_MFSK(CNR_per_bit_upper_puck, 2)

        # If we are going to pass the limit of error probability, place a puck
        # and then reset all of the upper puck stuff
        if probability_of_error > limit_probability_of_error \
            and probability_of_error_upper_puck > limit_probability_of_error:
            hf_number_of_pucks += 1

            # Record data
            placement_depths.append(eim.cryosphere_model_df.loc[d]['Depth (m)'])
            received_powers.append(received_power)
            noise_powers.append(noise_power)
            attenuations.append(attenuation)
            meter_distance_per_waves.append(meter_distance_per_wave)

            # Reset upper puck things

            lambda_s_t = c / (np.sqrt(epsilon_s_prime) * mm.low_band_f)

            T_A = eim.cryosphere_model_df.loc[d]['T_A Downwelling Low Band (K)']

            antenna_temperature = radiation_efficiency * matching_efficiency * T_A \
                + (1 - radiation_efficiency) * matching_efficiency * temperature_ice \
                    + (1 - matching_efficiency) * 290
            system_temperature = antenna_temperature + receiver_temperature
            noise_power_upper_puck = k * system_temperature * bandwidth

            meter_distance_per_wave = 0
            attenuation = 1

    recorded_data = pd.DataFrame({
        'Placement Depth (m)': placement_depths, 
        'Received Power (W)': received_power, 
        'Noise Power (W)': noise_power,
        'Attenuation': attenuations,
        'meter_distance_per_waves': meter_distance_per_waves})
    recorded_data.to_csv('recorded_data_low_band' + file_suffix + '.csv')
    eim.cryosphere_model_df.to_csv('cryosphere_model' + file_suffix + '.csv')

    return hf_number_of_pucks, uhf_number_of_pucks

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

class uhf_antenna():
    def directivity(self, T):
        directivity_in_100K_ice = 4.64 # dB
        directivity_in_273K_ice = 4.46 # dB
        m, b = math_funcs.linear_fit(
            100, math_funcs.db_2_power(directivity_in_100K_ice), 
            273, math_funcs.db_2_power(directivity_in_273K_ice))
        return m * T + b

    def radiation_efficiency(self, T):
        rad_eff_in_273K_ice = 0.228 # dB
        rad_eff_in_100K_ice = 0.223 # dB
        m, b = math_funcs.linear_fit(
            100, rad_eff_in_100K_ice, 
            273, rad_eff_in_273K_ice)
        return m * T + b
    
    def realized_gain(self, T):
        return self.directivity(T) * self.radiation_efficiency(T) * self.matching_efficiency

    matching_efficiency = 0.952
    carrier_frequency = 413e6 # Hz

class hf_antenna():
    directivity = math_funcs.db_2_power(1.73) # dB
    radiation_efficiency = 0.007
    matching_efficiency = 1
    carrier_frequency = 5.373e6 # Hz

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

        # Modify the epsilon s primes by the TVB
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
    hf_number_of_pucks, uhf_number_of_pucks =\
        evaluate_number_of_pucks_on_arbitrary_europa()
    print(f'HF Pucks {hf_number_of_pucks} and UHF pucks {uhf_number_of_pucks}')
    print("--- %s seconds ---" % (time.time() - start_time))
    