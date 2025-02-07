# Imports and plotting setups
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math

import math_funcs

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

#Define coldspace, jovian, gal radiation field for a mission 
T_A_DAM_low_band = 10**11 #K
T_A_DIM_high_band = 10**2 #K
T_A_gal_low_band = 10**7 # K
T_A_gal_high_band = T_A_DIM_high_band
T_B_coldspace = 2.7 #K

# Define a function to resample to the 1 deg find 
# grid to 10 deg course grid that the antenna pattern is using
class angle_grid:
    def __init__(self, dtheta, dphi):
        self.dtheta = dtheta
        self.dphi = dphi

        self.dtheta_rad = np.deg2rad(dtheta)
        self.dphi_rad = np.deg2rad(dphi)

        self.theta = np.arange(0, 180+dtheta, dtheta, dtype=int) # deg
        self.phi = np.arange(0, 360+dphi, dphi, dtype=int) # deg

        self.d_solid_angle = \
            np.sin(np.deg2rad(self.theta.flatten()))[:, None] \
                * self.dtheta_rad * self.dphi_rad

        self.phi_grid, \
            self.theta_grid = \
                np.meshgrid(self.phi, self.theta)
        
    def get_angle_grid(self):
        return self.theta, self.phi, \
            self.phi_grid, self.theta_grid

def deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A):

    theta, phi, \
        phi_grid, theta_grid  = ag.get_angle_grid()

    # Integrate the weighting pattern and the directivity 
    # of the antenna from the Cecconi paper
    sum_v = 0 
    sum_h = 0
    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        
        dsteradians = np.abs(np.sin(theta)) * ag.dtheta_rad * ag.dphi_rad
        
        sum_v += W_v[i] * (3/2) * (np.sin(theta))**2 * dsteradians
        sum_h += W_h[i] * (3/2) * (np.sin(theta))**2 * dsteradians

    T_Bv = W_v * 4 * np.pi * T_A / sum_v / 2 
    T_Bh = W_h * 4 * np.pi * T_A / sum_h / 2
    
    return T_Bv, T_Bh

def resample_course(T_B, return_grids=False):
    # TODO Magic numbers galore in this one... probably should parameterize this if I get a chance
    # Define the course theta and phi sampling grid
    theta_resample, phi_resample, \
        phi_grid_resample, theta_grid_resample = angle_grid(10, 10).get_angle_grid()

    # Resample and reweight the sampling grid points
    T_B_resample = np.zeros_like(theta_grid_resample, dtype=float).flatten()
    for i in np.arange(theta_grid_resample.size):
        theta = theta_grid_resample.flatten()[i]
        phi = phi_grid_resample.flatten()[i]

        theta_range = (theta + np.arange(-5, 5, 1)) % 180
        phi_range = (phi + np.arange(-5, 5, 1)) % 360
        dsteradian = 100
        for sub_theta in theta_range:
            for sub_phi in phi_range:
                T_B_resample[i] += T_B[sub_theta, sub_phi]

        T_B_resample[i] = T_B_resample[i] / dsteradian

    T_B_resample = T_B_resample.reshape(theta_grid_resample.shape)

    if return_grids:
        return T_B_resample, theta_resample, phi_resample, theta_grid_resample, phi_grid_resample
    else:
        return T_B_resample

# Brightness Field For the SubJovian point
def T_B_sub_jovian_point():
    # Define the theta and phi fine sampling grid
    ag = angle_grid(1, 1)
    theta, phi, \
        phi_grid, theta_grid  = ag.get_angle_grid()

    # Define the brightness grid with vertical and horizontal polarization
    T_Bv_low_band = np.zeros_like(theta_grid, dtype=float).flatten()
    T_Bh_low_band = np.zeros_like(theta_grid, dtype=float).flatten()
    T_Bv_high_band = np.zeros_like(theta_grid, dtype=float).flatten()
    T_Bh_high_band = np.zeros_like(theta_grid, dtype=float).flatten()

    # DAM radiation illumination 
    # from aural regions of jupiter visible only 1 degree
    # And only at the low band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta == 6:
            if phi == 0 or phi == 180:
                W_v[i] = 1 
                W_h[i] = 1 
    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_DAM_low_band)

    T_Bv_low_band += T_Bv
    T_Bh_low_band += T_Bh

    # DIM radiation illumination 
    # from jupiter itself directly
    # Only really visible in the high band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta <= 12:
            W_v[i] = 1 
            W_h[i] = 1 

    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_DIM_high_band)

    T_Bv_high_band += T_Bv
    T_Bh_high_band += T_Bh

    # Draw the galactic background
    # Low band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta <= 5: # directly overhead 
            if phi >= 0 and phi <= 360:
                W_v[i] = 0.5
                W_h[i] = 0.5

        if phi >= theta * (45 / 100) and phi <= theta * (55/100):
            W_v[i] = 0.5
            W_h[i] = 0.5

        if phi >= theta * (45 / 100) + 180 and phi <= theta * (55/100) + 180:
            W_v[i] = 0.5
            W_h[i] = 0.5

    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_gal_low_band)

    T_Bv_low_band += T_Bv
    T_Bh_low_band += T_Bh

    # Draw the galactic background
    # High band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta <= 5: # directly overhead 
            if phi >= 0 and phi <= 360:
                W_v[i] = 0.5
                W_h[i] = 0.5

    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_gal_high_band)

    T_Bv_high_band += T_Bv
    T_Bh_high_band += T_Bh

    # Addition of cold space background everywhere
    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        
        
        T_Bv_low_band[i] += 0.5 * T_B_coldspace
        T_Bh_low_band[i] += 0.5 * T_B_coldspace
        T_Bv_high_band[i] += 0.5 * T_B_coldspace
        T_Bh_high_band[i] += 0.5 * T_B_coldspace

    T_Bv_low_band = resample_course(T_Bv_low_band.reshape(theta_grid.shape))
    T_Bh_low_band = resample_course(T_Bh_low_band.reshape(theta_grid.shape))
    T_Bv_high_band = resample_course(T_Bv_high_band.reshape(theta_grid.shape))
    T_Bh_high_band = resample_course(T_Bh_high_band.reshape(theta_grid.shape))

    return T_Bv_low_band, T_Bh_low_band, T_Bv_high_band, T_Bh_high_band

# Brightness Field For the AntiJovian point
def T_B_anti_jovian_point():
    # Define the theta and phi fine sampling grid
    ag = angle_grid(1, 1)
    theta, phi, \
        phi_grid, theta_grid  = ag.get_angle_grid()

    # Define the brightness grid with vertical and horizontal polarization
    T_Bv_low_band = np.zeros_like(theta_grid, dtype=float).flatten()
    T_Bh_low_band = np.zeros_like(theta_grid, dtype=float).flatten()
    T_Bv_high_band = np.zeros_like(theta_grid, dtype=float).flatten()
    T_Bh_high_band = np.zeros_like(theta_grid, dtype=float).flatten()

    # DAM radiation illumination 
    # from aural regions of jupiter visible only 1 degree
    # And only at the low band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta == 174:
            if phi == 0 or phi == 180:
                W_v[i] = 1 
                W_h[i] = 1 
    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_DAM_low_band)

    T_Bv_low_band += T_Bv
    T_Bh_low_band += T_Bh

    # DIM radiation illumination 
    # from jupiter itself directly
    # Only really visible in the high band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta >= 168:
            W_v[i] = 1 
            W_h[i] = 1 

    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_DIM_high_band)

    T_Bv_high_band += T_Bv
    T_Bh_high_band += T_Bh

    # Draw the galactic background
    # Low band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta >= 45 and theta <= 75:
            if phi >= 0 and phi <= 180:
                W_v[i] = 1 
                W_h[i] = 1 

    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_gal_low_band)

    T_Bv_low_band += T_Bv
    T_Bh_low_band += T_Bh

    # Draw the galactic background
    # High band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta >= 45 and theta <= 75:
            if phi >= 0 and phi <= 180:
                W_v[i] = 1 
                W_h[i] = 1

    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_gal_high_band)

    T_Bv_high_band += T_Bv
    T_Bh_high_band += T_Bh

    # Addition of cold space background everywhere
    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        
        
        T_Bv_low_band[i] += 0.5 * T_B_coldspace
        T_Bh_low_band[i] += 0.5 * T_B_coldspace
        T_Bv_high_band[i] += 0.5 * T_B_coldspace
        T_Bh_high_band[i] += 0.5 * T_B_coldspace

    T_Bv_low_band = resample_course(T_Bv_low_band.reshape(theta_grid.shape))
    T_Bh_low_band = resample_course(T_Bh_low_band.reshape(theta_grid.shape))
    T_Bv_high_band = resample_course(T_Bv_high_band.reshape(theta_grid.shape))
    T_Bh_high_band = resample_course(T_Bh_high_band.reshape(theta_grid.shape))

    return T_Bv_low_band, T_Bh_low_band, T_Bv_high_band, T_Bh_high_band

# Brightness Field For the AntiOrbital point
def T_B_anti_orbital_point():
    # Define the theta and phi fine sampling grid
    ag = angle_grid(1, 1)
    theta, phi, \
        phi_grid, theta_grid  = ag.get_angle_grid()

    # Define the brightness grid with vertical and horizontal polarization
    T_Bv_low_band = np.zeros_like(theta_grid, dtype=float).flatten()
    T_Bh_low_band = np.zeros_like(theta_grid, dtype=float).flatten()
    T_Bv_high_band = np.zeros_like(theta_grid, dtype=float).flatten()
    T_Bh_high_band = np.zeros_like(theta_grid, dtype=float).flatten()

    # DAM radiation illumination 
    # from aural regions of jupiter visible only 1 degree
    # And only at the low band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta == 96 or theta == 84:
            if phi == 270:
                W_v[i] = 1 
                W_h[i] = 1 
    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_DAM_low_band)

    T_Bv_low_band += T_Bv
    T_Bh_low_band += T_Bh

    # DIM radiation illumination 
    # from jupiter itself directly
    # Only really visible in the high band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta >= 78 and theta <= 102:
            if phi >= 258 and phi <= 282:
                W_v[i] = 1 
                W_h[i] = 1 

    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_DIM_high_band)

    T_Bv_high_band += T_Bv
    T_Bh_high_band += T_Bh

    # Draw the galactic background
    # Low band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta >= 15 and theta <= 45:
            if phi >= 0 and phi <= 180:
                W_v[i] = 1 
                W_h[i] = 1 

    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_gal_low_band)

    T_Bv_low_band += T_Bv
    T_Bh_low_band += T_Bh

    # Draw the galactic background
    # High band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta >= 15 and theta <= 45:
            if phi >= 0 and phi <= 180:
                W_v[i] = 1 
                W_h[i] = 1

    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_gal_high_band)

    T_Bv_high_band += T_Bv
    T_Bh_high_band += T_Bh

    # Addition of cold space background everywhere
    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        
        
        T_Bv_low_band[i] += 0.5 * T_B_coldspace
        T_Bh_low_band[i] += 0.5 * T_B_coldspace
        T_Bv_high_band[i] += 0.5 * T_B_coldspace
        T_Bh_high_band[i] += 0.5 * T_B_coldspace

    T_Bv_low_band = resample_course(T_Bv_low_band.reshape(theta_grid.shape))
    T_Bh_low_band = resample_course(T_Bh_low_band.reshape(theta_grid.shape))
    T_Bv_high_band = resample_course(T_Bv_high_band.reshape(theta_grid.shape))
    T_Bh_high_band = resample_course(T_Bh_high_band.reshape(theta_grid.shape))

    return T_Bv_low_band, T_Bh_low_band, T_Bv_high_band, T_Bh_high_band

if __name__ == "__main__":
    ag = angle_grid(1, 1)
    theta, phi, \
        phi_grid, theta_grid  = ag.get_angle_grid()

    # Draw the galactic background
    # Low band
    W_v = np.zeros_like(theta_grid, dtype=float).flatten()
    W_h = np.zeros_like(theta_grid, dtype=float).flatten()

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]
        if theta >= 105 and theta <= 135:
            if phi >= 180 and phi <= 360:
                W_v[i] = 0.5
                W_h[i] = 0.5

    T_Bv, T_Bh = deconvolve_Cecconi_T_A_to_T_B(ag, W_h, W_v, T_A_gal_low_band)

    T_A = 0

    for i in np.arange(theta_grid.size):
        theta = theta_grid.flatten()[i]
        phi = phi_grid.flatten()[i]

        dsteradians = np.abs(np.sin(theta)) * ag.dtheta_rad * ag.dphi_rad
        
        T_A += T_Bv[i] * (3/2) * (np.sin(theta))**2 * dsteradians
        T_A += T_Bh[i] * (3/2) * (np.sin(theta))**2 * dsteradians

    T_A *= (1 / (4* np.pi))

    print(f'Percent error in deconvolve, convolve method: {(T_A - T_A_gal_low_band)*100}%')
    print(f'{T_A}, and {T_A_gal_low_band}')

    for func, title in [
        (T_B_sub_jovian_point, 'Sub-Jovian Point'), 
        (T_B_anti_jovian_point, 'Anti-Jovian Point'),
        (T_B_anti_orbital_point, 'Anti-Orbital Point')]:

        T_Bv_low_band, T_Bh_low_band, \
            T_Bv_high_band, T_Bh_high_band \
             = func()
        
        # The course grid
        theta, phi, \
            phi_grid, theta_grid = angle_grid(10, 10).get_angle_grid()

        theta_grid_rad = np.deg2rad(theta_grid)
        phi_grid_rad = np.deg2rad(phi_grid)

        # Plot only the Northern Hemisphere of the brightness temperature
        U = (np.sin(theta_grid_rad[0:9]) * np.cos(phi_grid_rad[0:9])).flatten()
        V = (np.sin(theta_grid_rad[0:9]) * np.sin(phi_grid_rad[0:9])).flatten()

        plt.figure()
        plt.scatter(U, V, c=10 * np.log10(T_Bv_low_band.reshape(theta_grid_rad.shape)[0:9].flatten())
            , cmap='viridis', vmin=0, vmax=200)
        plt.colorbar(label='Brightness Temperature (dBK)')
        plt.xlabel('U')
        plt.ylabel('V')
        plt.title('Northern UV Projection of Vertical Brightness Temperature Low Band')
        plt.suptitle(title)
        plt.savefig(title + ' Northern UV Projection of Vertical Brightness Temperature Low Band.pdf')

        # Plot only the Northern Hemisphere of the brightness temperature
        U = (np.sin(theta_grid_rad[0:9]) * np.cos(phi_grid_rad[0:9])).flatten()
        V = (np.sin(theta_grid_rad[0:9]) * np.sin(phi_grid_rad[0:9])).flatten()

        plt.figure()
        plt.scatter(U, V, c=10 * np.log10(T_Bv_high_band.reshape(theta_grid_rad.shape)[0:9].flatten())
            , cmap='viridis', vmin=0, vmax=200)
        plt.colorbar(label='Brightness Temperature (dBK)')
        plt.xlabel('U')
        plt.ylabel('V')
        plt.title('Northern UV Projection of Vertical Brightness Temperature High Band')
        plt.suptitle(title)
        plt.savefig(title + ' Northern UV Projection of Vertical Brightness Temperature High Band.pdf')



