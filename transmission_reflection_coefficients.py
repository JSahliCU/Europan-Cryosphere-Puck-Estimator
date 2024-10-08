import numpy as np
from scipy.constants import (
    epsilon_0,   # Permittivity of free space (vacuum)
    mu_0,        # Permeability of free space (vacuum)
    c,           # Speed of light in vacuum
)
from scipy.special import loggamma, factorial, gamma
from decimal import Decimal
import sys

def transmissivity_fBm(
        f, epsilon_i_prime, epsilon_t_prime, sigma_A, H, 
        theta_i, theta_t, phi_t, bistatic_polarization,
        tolerance = 1e-5,
        decimal_precision = int(1e3),
        max_n_convergence = 10.0,
        debug = False
        ):
    """
    Based on the method presented by Franceschetti and Riccio for scattering from fractional brownian surfaces
    This function calculates the transmissivity coefficient for an fBm 
    surface described by 2D allan variance, Hurst coefficient at the passedbistatic polarization

    Giorgio Franceschetti, Daniele Riccio, 
    “Chapter 6 - Scattering from Fractional 
    Brownian Surfaces: Physical-Optics Solution” 
    in Scattering, Natural Surfaces, and Fractals, 2006

    Args:
        f (float): Frequency
        epsilon_i_prime (float): permittivity of the incident medium
        epsilon_t_prime (float): permittivity of the medium the wave is transmitted to
        sigma_A (float): 2D Allan variance of the surface
        H (float): Hurst exponent
        theta_i (float): incident theta angle
        theta_t (float): transmission theta angle
        phi_t (float): transmission phi angle
        bistatic_polarization (str): 'vv', 'hv', 'vh' or 'hh'
        tolerance (float, optional): Convergence criteria: If the difference between subsequent 
            series elements is less than tolerance, the series is assumed to converge. Defaults to 1e-5.
        decimal_precision (int, optional): The amount of decimal precision to use for the series elements. Defaults to int(1e3).
        max_n_convergence (float, optional): Maximum number of series elemnts. Defaults to 10.0.
        debug (bool, optional): Enables debug mode. Defaults to False.

    Returns:
        float: transmissivity coefficient for an fBm surface described by sigma_A, H at the bistatic polarization
    """

    # Evaluate constants
    lambda_i = c / (f * np.sqrt(epsilon_i_prime)) 
    lambda_t = c / (f * np.sqrt(epsilon_t_prime)) 
    k_i = 2 * np.pi / lambda_i
    k_t = 2 * np.pi / lambda_t

    eta_xy = np.sqrt((k_t * np.sin(theta_t) * np.cos(phi_t) - k_i * np.sin(theta_i))**2 \
                    + (k_t * np.sin(theta_t) * np.sin(phi_t))**2)
    eta_z = k_i * np.cos(theta_i) - k_t * np.cos(theta_t)

    # Evaluate the fBm series
    sum, terms_to_converge, accuracy_overflow = _evaluate_fBm_series(
        sigma_A, H, 
        eta_xy, eta_z,
        tolerance,
        decimal_precision,
        max_n_convergence,
        debug
    )

    ## Calculate the transmissivity
    eta_i = np.sqrt(mu_0 / (epsilon_i_prime * epsilon_0))
    eta_t = np.sqrt(mu_0 / (epsilon_t_prime * epsilon_0))

    N_ab = bistatic_transmission_coefficients(
        epsilon_i_prime, epsilon_t_prime, theta_i, 
        theta_t, phi_t, bistatic_polarization
        )

    gamma_ab_t = (k_t**2 / np.cos(theta_i)) * (eta_i / eta_t) * np.abs(N_ab)**2 * (1 / (2*H)) * float(sum)

    if debug:
        return gamma_ab_t, sum, terms_to_converge, accuracy_overflow
    else:
        return gamma_ab_t
    

def reflectivity_fBm(
        f, epsilon_i_prime, epsilon_t_prime, sigma_A, H, 
        theta_i, theta_s, phi_s, bistatic_polarization,
        tolerance = 1e-5,
        decimal_precision = int(1e3),
        max_n_convergence = 10.0,
        debug = False
        ):
    """
    Based on the method presented by Franceschetti and Riccio for scattering from fractional brownian surfaces
    This function calculates the reflectivity coefficient for an fBm 
    surface described by 2D allan variance, Hurst coefficient at the passedbistatic polarization

    Giorgio Franceschetti, Daniele Riccio, 
    “Chapter 6 - Scattering from Fractional 
    Brownian Surfaces: Physical-Optics Solution” 
    in Scattering, Natural Surfaces, and Fractals, 2006

    Args:
        f (float): Frequency
        epsilon_i_prime (float): permittivity of the incident medium
        epsilon_t_prime (float): permittivity of the scattering medium
        sigma_A (float): 2D Allan variance of the surface
        H (float): Hurst exponent
        theta_i (float): incident theta angle
        theta_t (float): scattering theta angle
        phi_t (float): scattering phi angle
        bistatic_polarization (str): 'vv', 'hv', 'vh' or 'hh'
        tolerance (float, optional): Convergence critera: If the difference between 
            subsequent series elements is less than tolerance, the series is assumed to converge. Defaults to 1e-5.
        decimal_precision (int, optional): The amount of decimal precision to use for the series elements. Defaults to int(1e3).
        max_n_convergence (float, optional): Maximum number of series elemnts. Defaults to 10.0.
        debug (bool, optional): Enables debug mode. Defaults to False.

    Returns:
        float: reflectivity coefficient for an fBm surface described by sigma_A, H at the bistatic polarization
    """
    # Evaluate constants
    lambda_i = c / (f * np.sqrt(epsilon_i_prime)) 
    k_i = 2 * np.pi / lambda_i

    eta_xy = np.sqrt((k_i * np.sin(theta_s) * np.cos(phi_s) - k_i * np.sin(theta_i))**2 \
                    + (k_i * np.sin(theta_s) * np.sin(phi_s))**2)
    eta_z = k_i * np.cos(theta_i) + k_i * np.cos(theta_s)

    if theta_i >= np.pi/2 or theta_s >= np.pi/2:
        sum = 0
        terms_to_converge = 0
        accuracy_overflow = False
        gamma_ab_s = 0
    else:
        # Evaluate the fBm series
        sum, terms_to_converge, accuracy_overflow = _evaluate_fBm_series(
            sigma_A, H, 
            eta_xy, eta_z,
            tolerance,
            decimal_precision,
            max_n_convergence,
            debug
        )

        ## Calculate the reflectivity
        F_ab = bistatic_scattering_coefficients(
            epsilon_i_prime, epsilon_t_prime, theta_i, 
            theta_s, phi_s, bistatic_polarization)

        gamma_ab_s = (k_i**2 / np.cos(theta_i)) * np.abs(F_ab)**2 * (1 / (2*H)) * float(sum)

    if debug:
        return gamma_ab_s, sum, terms_to_converge, accuracy_overflow
    else:
        return gamma_ab_s

def fresnel_coefficients(n_i, n_t, theta_i):
    """Calculation for fresnel reflection coefficients

    Args:
        n_i (float): index of refrection of incident medium
        n_t (float): index of refrection of the scattering medium
        theta_i (float): incident angle of the wave front measured from the normal of medium interface

    Returns:
        float, float: Vertical and horizontal fresnel coefficients, respectively
    """
    sin_theta_t = n_i * np.sin(theta_i) / n_t
    # Case of total internal reflection
    if sin_theta_t > 1:
        R_v = 1
        R_h = 1
    else:
        theta_t = np.arcsin(sin_theta_t)
        R_v = (n_i * np.cos(theta_i) - n_t * np.cos(theta_t)) \
                / (n_i * np.cos(theta_i) + n_t * np.cos(theta_t))

        R_h = (n_i * np.cos(theta_t) - n_t * np.cos(theta_i)) \
            / (n_i * np.cos(theta_t) + n_t * np.cos(theta_i))
    return R_v, R_h

def bistatic_scattering_coefficients(
    epsilon_i_prime, epsilon_t_prime, theta_i, 
    theta_s, phi_s, bistatic_polarization):
    """
    Bistatic fresnel scattering coefficients
    Tsang, Leung, Kong, Jin Au, Shin, Robert T. 
    “Ch 2.6. Scattering and Emission by Random Rough Surfaces” 
    in Theory of Microwave Remote Sensing, 1985.
    Args:
        epsilon_i_prime (float): permittivity of the incident medium
        epsilon_t_prime (float): permittivity of the scattering medium
        theta_i (float): incident angle of the wave front measured from the normal of medium interface
        theta_s (float): scattering theta angle of the wave front measured from the normal of medium interface
        phi_s (float): scattering phi angle of the wave front measured from incident phi angle
        bistatic_polarization (str): 'vv', 'hv', 'vh' or 'hh'

    Returns:
        float : bistatic fresnel scattering coefficients
    """

    n_i = np.sqrt(epsilon_i_prime * epsilon_0 * mu_0)
    n_t = np.sqrt(epsilon_t_prime * epsilon_0 * mu_0)

    R_v, R_h = fresnel_coefficients(n_i, n_t, theta_i)

    if bistatic_polarization == 'hh':
        F_ab = ((1 - R_h) * np.cos(theta_i) - (1 + R_h) * np.cos(theta_s)) * np.cos(phi_s)
    elif bistatic_polarization == 'hv':
        F_ab = ((1 + R_h) - (1 - R_h) * np.cos(theta_i) * np.cos(theta_s)) * np.sin(phi_s)
    elif bistatic_polarization == 'vh':
        F_ab = ((1 + R_v) * np.cos(theta_i) * np.cos(theta_s) - (1 - R_v)) * np.sin(phi_s)
    elif bistatic_polarization == 'vv':
        F_ab = (-1 * (1 - R_v) * np.cos(theta_s) + (1 + R_v) * np.cos(theta_i)) * np.cos(phi_s)
    else:
        raise(ValueError('bistatic polarization must be \'vv\',  \'vh\',  \'hv\',  or \'hh\''))
    
    return F_ab

def bistatic_transmission_coefficients(
    epsilon_i_prime, epsilon_t_prime, theta_i, 
    theta_t, phi_t, bistatic_polarization):
    """
    Bistatic Fresnel transmission coefficients
    Tsang, Leung, Kong, Jin Au, Shin, Robert T. 
    “Ch 2.6. Scattering and Emission by Random Rough Surfaces” 
    in Theory of Microwave Remote Sensing, 1985.
    Args:
        epsilon_i_prime (float): permittivity of the incident medium
        epsilon_t_prime (float): permittivity of the transmission medium
        theta_i (float): incident angle of the wave front measured from the normal of medium interface
        theta_t (float): transmission theta angle of the wave front measured from the normal of medium interface
        phi_t (float): transmission phi angle of the wave front measured from incident phi angle
        bistatic_polarization (str): 'vv', 'hv', 'vh' or 'hh'

    Returns:
        float : bistatic Fresnel scattering coefficients
    """
    eta_i = np.sqrt(mu_0 / (epsilon_i_prime * epsilon_0))
    eta_t = np.sqrt(mu_0 / (epsilon_t_prime * epsilon_0))
    n_i = np.sqrt(epsilon_i_prime * epsilon_0 * mu_0)
    n_t = np.sqrt(epsilon_t_prime * epsilon_0 * mu_0)

    R_v, R_h = fresnel_coefficients(n_i, n_t, theta_i)

    if bistatic_polarization == 'hh':
        N_ab = ((1 + R_h) * np.cos(theta_t) + (eta_t / eta_i) * (1 - R_h) * np.cos(theta_i)) * np.cos(phi_t)
    elif bistatic_polarization == 'hv':
        N_ab = (-1 * (1 + R_h) - (eta_t / eta_i) * (1 - R_h) * np.cos(theta_i) * np.cos(theta_t)) * np.sin(phi_t)
    elif bistatic_polarization == 'vh':
        N_ab = ((eta_t / eta_i) * (1 + R_v) - (1 - R_v) * np.cos(theta_i) * np.cos(theta_t)) * np.sin(phi_t)
    elif bistatic_polarization == 'vv':
        N_ab = ((eta_t / eta_i) * (1 + R_v) + (1 - R_v) * np.cos(theta_i)) * np.cos(phi_t)
    else:
        raise(ValueError('bistatic polarization must be \'vv\',  \'vh\',  \'hv\',  or \'hh\''))

    return N_ab

def _evaluate_fBm_series(
        sigma_A, H, 
        eta_xy, eta_z,
        tolerance = 1e-5,
        decimal_precision = int(1e3),
        max_n_convergence = 10.0,
        debug = False):
    
    """
    Based on the method presented by Franceschetti and Riccio for scattering 
    from fractional brownian surfaces
    This function calculates the series for a persistent fBm 
    surface described by 2D allan variance, Hurst coefficient at the 
    passed bistatic polarization

    Currently only supports persistent fBm

    Giorgio Franceschetti, Daniele Riccio, 
    “Chapter 6 - Scattering from Fractional 
    Brownian Surfaces: Physical-Optics Solution” 
    in Scattering, Natural Surfaces, and Fractals, 2006

    Args:
        sigma_A (float): 2D Allan variance of the surface
        H (float): Hurst exponent
        eta_xy (float): wave vector in the xy direction
        eta_z (float): wave vector in the z direction
        tolerance (float, optional): Convergence criteria: If the difference between 
            subsequent series elements is less than tolerance, the series is assumed to converge. Defaults to 1e-5.
        decimal_precision (int, optional): The amount of decimal precision to use for the series elements. Defaults to int(1e3).
        max_n_convergence (float, optional): Maximum number of series elements. Defaults to 10.0.
        debug (bool, optional): Enables debug mode. Defaults to False.

    Returns:
        float: fBm Series output
    """

    if H < 0.5:
        raise(ValueError('This series is only valid for H >= 1/2'))

    # Evaluate the series S at 0
    n = 0

    first_fraction = ((-1) ** n) / (2**(2*n) * (1.0)**2)
    second_fraction = (eta_xy)**(2*n) / (0.5 * eta_z**2 * sigma_A**2)**((n+1)/H)
    third_factor = gamma((n + 1) / H)

    sum = Decimal(first_fraction*second_fraction*third_factor)
    delta_sum_list = [np.log(first_fraction*second_fraction*third_factor)]
    old_log_factorial = np.log(1.0)
    accuracy_overflow = False

    # Evaluate the series S from 1 to max_n_convergence
    # or until the terms is less than tolerance
    for n in np.arange(1.0, max_n_convergence + 1.0, 1):

        new_log_factorial = 2 * np.log(n) + old_log_factorial
        old_log_factorial = new_log_factorial
        kernel =  -2 * n * np.log(2) \
            - new_log_factorial \
                + 2 * n * np.log(eta_xy) \
                    - ((n+1) / H)*np.log(0.5 * eta_z**2 * sigma_A**2) \
                        + loggamma((n + 1) / H)
        
        delta_sum = Decimal(np.e)**Decimal(kernel)
        try:
            sum += Decimal((-1**n)) * delta_sum
        except(BaseException):
            print(sigma_A, H, 
                eta_xy, eta_z, sum, n, delta_sum)
            sys.exit(-1)
            
        delta_sum_list.append(kernel)

        if kernel > decimal_precision :
            accuracy_overflow = True

        if n != 0:   
            if tolerance > delta_sum:
                break
    
    # Mask the sum if it exceeded the max n convergence, 
    # or if the decimal precision was not high enough
    terms_to_converge = n

    if terms_to_converge >= max_n_convergence:
        sum = np.nan
        
    if accuracy_overflow:
        sum = np.nan

    return sum, terms_to_converge, accuracy_overflow

def transmissivity_gaussian(
    f, epsilon_i_prime, epsilon_t_prime, sigma_h, correlation_length, 
    theta_i, theta_t, phi_t, bistatic_polarization,
    tolerance = 1e-5,
    decimal_precision = int(1e3),
    max_n_convergence = 10.0,
    debug = False
    ):
    """
    Calculates the bistatic transmissivity coefficients of Gaussian Surface
    Tsang, Leung, Kong, Jin Au, Shin, Robert T. 
    “Ch 2.6. Scattering and Emission by Random Rough Surfaces” 
    in Theory of Microwave Remote Sensing, 1985.
    Args:
        f (float): frequency to evaluate
        epsilon_i_prime (float): permittivity of the incident medium
        epsilon_t_prime (float): permittivity of the transmission medium
        sigma_h (float): variance of the surface height
        correlation_length (float): correlation length of surface
        theta_i (float): incident angle
        theta_t (float): transmitted angle
        phi_t (float): transmitted phi angle, measured from the incident phi angle
        bistatic_polarization (float): Can be 'vv', 'vh', 'hv', or 'hh'
        tolerance (float, optional): Tolerance to terminate series. Defaults to 1e-5.
        decimal_precision (int, optional): Number of digits in series evaluation. Defaults to int(1e3).
        max_n_convergence (float, optional): Maximum number of series terms before termination. Defaults to 10.0.
        debug (bool, optional): Turns on debug information. Defaults to False.

    Returns:
        float: bistatic transmissivity coefficient of gaussian surface
    """

    lambda_i = c / (f * np.sqrt(epsilon_i_prime)) 
    lambda_t = c / (f * np.sqrt(epsilon_t_prime)) 
    k_i = 2 * np.pi / lambda_i
    k_t = 2 * np.pi / lambda_t

    eta_xy = np.sqrt((k_t * np.sin(theta_t) * np.cos(phi_t) - k_i * np.sin(theta_i))**2 \
                    + (k_t * np.sin(theta_t) * np.sin(phi_t))**2)
    eta_z = k_i * np.cos(theta_i) - k_t * np.cos(theta_t)

    # Evaluate the fBm series
    sum, terms_to_converge, accuracy_overflow = _evaluate_gaussian_series(
        sigma_h, correlation_length, 
        eta_xy, eta_z,
        tolerance,
        decimal_precision,
        max_n_convergence,
        debug
    )

    ## Calculate the transmissivity
    eta_i = np.sqrt(mu_0 / (epsilon_i_prime * epsilon_0))
    eta_t = np.sqrt(mu_0 / (epsilon_t_prime * epsilon_0))

    N_ab = bistatic_transmission_coefficients(
        epsilon_i_prime, epsilon_t_prime, theta_i, 
        theta_t, phi_t, bistatic_polarization
        )

    gamma_ab_t = (k_t**2 / (4 * np.cos(theta_i))) \
        * (eta_i / eta_t) * np.abs(N_ab)**2 * float(sum)

    if debug:
        return gamma_ab_t, sum, terms_to_converge, accuracy_overflow
    else:
        return gamma_ab_t

def reflectivity_gaussian(
    f, epsilon_i_prime, epsilon_t_prime, sigma_h, correlation_length, 
    theta_i, theta_s, phi_s, bistatic_polarization,
    tolerance = 1e-5,
    decimal_precision = int(1e3),
    max_n_convergence = 10.0,
    debug = False
    ):
    """
    Calculates the bistatic reflectivity coefficients of Gaussian Surface
    Tsang, Leung, Kong, Jin Au, Shin, Robert T. 
    “Ch 2.6. Scattering and Emission by Random Rough Surfaces” 
    in Theory of Microwave Remote Sensing, 1985.

    Args:
        f (float): frequency to evaluate
        epsilon_i_prime (float): permittivity of the incident medium
        epsilon_t_prime (float): permittivity of the transmission medium
        sigma_h (float): variance of the surface height
        correlation_length (float): correlation length of surface
        theta_i (float): incident angle
        theta_s (float): scattered angle
        phi_s (float): scattered angle phi measured from the incident angle phi
        bistatic_polarization (float): Can be 'vv', 'vh', 'hv', or 'hh'
        tolerance (float, optional): Tolerance to terminate series. Defaults to 1e-5.
        decimal_precision (int, optional): Number of digits in series evaluation. Defaults to int(1e3).
        max_n_convergence (float, optional): Maximum number of series terms before termination. Defaults to 10.0.
        debug (bool, optional): Turns on debug information. Defaults to False.

    Returns:
        float: bistatic reflectivity coefficient of gaussian surface
    """

    # Evaluate constants
    lambda_i = c / (f * np.sqrt(epsilon_i_prime)) 
    k_i = 2 * np.pi / lambda_i

    eta_xy = np.sqrt((k_i * np.sin(theta_s) * np.cos(phi_s) - k_i * np.sin(theta_i))**2 \
                    + (k_i * np.sin(theta_s) * np.sin(phi_s))**2)
    eta_z = k_i * np.cos(theta_i) + k_i * np.cos(theta_s)

    # Evaluate the fBm series
    sum, terms_to_converge, accuracy_overflow = _evaluate_gaussian_series(
        sigma_h, correlation_length, 
        eta_xy, eta_z,
        tolerance,
        decimal_precision,
        max_n_convergence,
        debug
    )

    ## Calculate the reflectivity
    F_ab = bistatic_scattering_coefficients(
        epsilon_i_prime, epsilon_t_prime, theta_i, 
        theta_s, phi_s, bistatic_polarization)
    
    gamma_ab_s = (k_i**2 / (4 * np.cos(theta_i))) \
        * np.abs(F_ab)**2 * float(sum)

    if debug:
        return gamma_ab_s, sum, terms_to_converge, accuracy_overflow
    else:
        return gamma_ab_s

def _evaluate_gaussian_series(
        sigma_h, correlation_length,  
        eta_xy, eta_z,
        tolerance = 1e-5,
        decimal_precision = int(1e3),
        max_n_convergence = 10.0,
        debug = False):
    """
    Evaluates the series for the Gaussian surface 
    calculation of transmission and reflection coefficients
    Tsang, Leung, Kong, Jin Au, Shin, Robert T. 
    “Ch 2.6. Scattering and Emission by Random Rough Surfaces” 
    in Theory of Microwave Remote Sensing, 1985.

    Args:
        sigma_h (float): variance of the surface height
        correlation_length (float): correlation length of surface
        eta_xy (float): wave vector in the xy plane
        eta_z (float): wave vector in the z direction
        tolerance (float, optional): Tolerance to terminate series. Defaults to 1e-5.
        decimal_precision (int, optional): Number of digits in series evaluation. Defaults to int(1e3).
        max_n_convergence (float, optional): Maximum number of series terms before termination. Defaults to 10.0.
        debug (bool, optional): Turns on debug information. Defaults to False.

    Returns:
        float: result of series
    """

    sum = Decimal(0)
    delta_sum_list = []
    old_log_factorial = np.log(1.0)
    accuracy_overflow = False

    # Evaluate the series S from 1 to max_n_convergence
    # or until the terms is less than tolerance
    for n in np.arange(1.0, max_n_convergence + 1.0, 1):
        new_log_factorial = np.log(n) + old_log_factorial
        old_log_factorial = new_log_factorial
        
        kernel =  -1 * new_log_factorial + np.log(correlation_length**2 / n) \
            + n * np.log(eta_z**2 * sigma_h**2) \
                - eta_xy **2 * correlation_length**2 / (4 * n) \
                    + sigma_h**2 * eta_z**2
        
        delta_sum = Decimal(np.e)**Decimal(kernel)
        sum += delta_sum
        delta_sum_list.append(kernel)
        
        if kernel > decimal_precision :
            accuracy_overflow = True

        if n != 0:   
            if tolerance > delta_sum:
                break
    
    # Mask the sum if it exceeded the max n convergence, 
    # or if the decimal precision was not high enough
    terms_to_converge = n

    if terms_to_converge >= max_n_convergence:
        sum = np.nan
        
    if accuracy_overflow:
        sum = np.nan

    return sum, terms_to_converge, accuracy_overflow