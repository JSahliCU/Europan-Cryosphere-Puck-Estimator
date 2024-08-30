#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
import EuropaMCdefinitions as EMC
import matplotlib as mpl
from matplotlib import pyplot as plt 
from scipy import interpolate
from scipy import stats
import scipy.stats as st
from scipy.stats import spearmanr
from scipy.stats import truncnorm

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Written by Samuel M. Howell for 
"The likely thickness of Europa’s icy shell"
samuel.m.howell@jpl.nasa.gov

(c) 2021 California Institute of Technology. All rights Reserved.

This software probabalistically models the thickness of Europa's ice
shell according to a Monte Carlo operation applied to a steady state
heat flux balance. See Howell (2021) for values and distributions
used here. Caution: Significant alteration may be required for the
correct application of this code to other bodies or pronlems.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
STEP 0: INPUTS AND COSNTANTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


" Input Parameters "
N = int(1e7) # Number of simulations
doPlots = 0 # 1 = make and save pdf dpPlots. 0 = off.
doStats = 0 # 1 = redo correlations (takes a long time). 0 = off.

" Physical Constants "
R = 8.31 # Universal gas constant [J/mol MC.K]
UGC = 6.67408e-11 # Universal Graviational constant [m^3/ks^2]

" Body Constants "
g = 1.315 # Acceleration due to gravity [m/s^2]
m_E = 4.80e22 # Mass of Europa [kg]
r_E = 1561e3 # Radius of Europa [m]
SA_Europa = 4 * np.pi * r_E**2 # Surface area of Europa [m^2]
omega = 2.047e-5  # Europan orbital frequency [1/s]
n_Europa = 2.0443e-5 # Orbital accelertation [1/s]
e_Europa = 0.0101 # Europa Orbital Eccentricity

" Ice Constants "
T_m0 = 273 # Melting temperature of pure ice Ih
rho0 = 917.0 # Melting temperature density of ice [kg/m^3]
Cp = 2000.0 # Heat capacity of ice [J/kg MC.K]
alpha = 1.6e-4 # Thermal expansivity of ice [K^-1]
d_Vm = 1.97e-5 # Molar volume of ice [m^3] (Fletcher 1970)







"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
STEP 1: CALCULATE THE INTERIOR HEAT FLUX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
print('\n\n\nEuropas are Under Construction\n')
print('Sampling Interior Heat Flux')

# Generate arrays
MC = EMC.MCarrays(N)

" Surface Temperature "
T_s_mean = 104.0 # Surface mean equitorial temeprature [MC.K]
T_s_un = 7.0 # Surface temperature stddev [MC.K]
T_s_sig = 1.0 # M sigma uncertainty
# Sample normal distribution
MC.T_s = np.random.normal(T_s_mean, T_s_un/T_s_sig,MC.N) # Surface Temperature [MC.K]


" H2O Layer Thickness "
D_H2O_mean = 127e3 # Mean full thickness of H2O layer [km]
D_H2O_un   = 21e3 # Uncertainty in thickness [km]
D_H2O_sig  = 1.0 # M sigma uncertainty
# Sample normal distribution
MC.D_H2O = np.random.normal(D_H2O_mean, D_H2O_un/D_H2O_sig,MC.N) #H2O Layer thickness [km]


" Radiogenic heating "
# First find spefic heat generation rate based on material source uncertainty
Q_ir_mean = 4.5e-12 # Mean specific heat generation rate [W/kg] 
Q_ir_un = 1.0e-12 # Uncertainty [W/kg] 
Q_ir_sig = 1.0 # M sigma uncertainty 
# Sample normal distribution
MC.Q_ir = np.random.normal(Q_ir_mean, Q_ir_un/Q_ir_sig, MC.N) # Specific heat generation rate [W/kg] 
MC = EMC.remove(MC,[MC.Q_ir < 0])

# Get radiogenic heat production
MC.q_ir = MC.Q_ir * m_E / (4.0 * np.pi * (r_E)**2) # Radiogenic heating rate [W/m^2]

# Dissipated power after Behounkova [2020]
Pi_mean = np.log10(100e9) # Dissipated power mean [w]
Pi_un = 1 # Uncertainty in orders of magnitude
Pi_sig = 3 # M sigma uncertainty 
# Sample normal distribution
MC.Pi = 10**np.random.normal(Pi_mean, Pi_un/Pi_sig, MC.N) # Grain size [m]

" Full tidal heat dissipation [W/m^2] "
MC.q_it = MC.Pi/ SA_Europa

" Summed interior heat generation "
MC.q_i = MC.q_ir + MC.q_it # Total heat generation from the interior evaluated at the surface [W/m^2]



"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
STEP 2: CALCULATE THE THERMAL PROPERTIES OF THE CONDUCTIVE LAYER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
print('Building conductive layer')


" Salt Eutectic Constants "
dT_m = 21 # Max deviation from T_m0 due to salts (Assuming NaCl Eutectic)
f_thresh = 0.22 # eutectic composition (Assuming NaCl Eutectic)

" Salt fraction "
f_s_mean = 0.03 # Mean salinity of ice shell
f_s_un = 1 # Order of magnitude Uncertainty
f_s_sig = 3 # M sigma uncertainty 
# Sample lognormal distribution
MC.f_s = 10**np.random.normal(np.log10(f_s_mean), f_s_un/f_s_sig, MC.N)  # Salt fraction
# Watch for non-physical extreme outliers
MC = EMC.remove(MC,[MC.f_s>f_thresh])

" Melting Temperature "
MC.T_m = T_m0 - dT_m * MC.f_s/f_thresh # Melting temperature [MC.K]

" Porosity "
phi_min = 0.0 # Minimum poroisty
phi_max = 0.3 # Maximum porosity
# Sample uniform distribution
MC.phi = np.random.uniform(phi_min, phi_max,MC.N) # Porosity (as volume fraction)


" Porosity Curing Temperature "
T_phi_mean = 150.0 # Lowest temp. where porosity cures on the timescale of the surface age
T_phi_un = 20 # Uncertainty
T_phi_sig = 3 # M sigma uncertainty 
# Sample normal distribution
MC.T_phi = np.random.normal(T_phi_mean, T_phi_un/T_phi_sig, MC.N) # Porosity curing temp. [MC.K]
# Watch for non-physical extreme outliers
MC = EMC.remove(MC,[MC.T_phi>MC.T_m])
MC = EMC.remove(MC,[MC.T_phi<MC.T_s])


" Convective core temperature "
# Note that some rheological properties are required early to get Tc
D0_var = 0.033 # Variance in prefactors (Ramsier 1967b)
Q_var = 0.05 # Variance in activation energy (Ramsier 1967b)

# Volume diffusion coefficient (Goldsby and Kohlstedt, 2001; Ramesier 1967b)
D0v_mean = 9.1e-4 # Volume Diffusion prefactor mean [m^2/s]
MC.D0v = np.random.normal(D0v_mean, D0_var*D0v_mean, MC.N) # Volume Diffusion prefactor [m^2/s]
Qv_mean = 59.4e3 # Volume diffusion activation energy mean [J/mol]
MC.Qv = np.random.normal(Qv_mean, Q_var*Qv_mean, MC.N) # Volume diffusion activation energy [J/mol]

# Grain boundary diffusion coefficient (Goldsby and Kohlstedt, 2001; Ramesier 1967b)
D0b_mean = 8.4e-4 # Boundary Diffusion prefactor mean [m^2/s]
MC.D0b = np.random.normal(D0b_mean, D0_var*D0b_mean, MC.N) # Boundary Diffusion prefactor [m^2/s]
Qb_mean = 49.0e3 # Boundary diffusion activation energy mean [J/mol]
MC.Qb = np.random.normal(Qb_mean, Q_var*Qb_mean, MC.N) # Boundary diffusion activation energy [J/mol]

# Now get Tc
MC.T_c = (np.sqrt(4*MC.T_m*(R/MC.Qv)+1)-1)/(2*(R/MC.Qv))
# Watch for non-physical extreme outliers
MC = EMC.remove(MC,[MC.T_c<MC.T_phi])
MC = EMC.remove(MC,[MC.T_c>MC.T_m])

" Conductive basal temperature "
MC.T_cond_base = 2*MC.T_c - MC.T_m; # Temp at base of conductive layer [MC.K]


" Calculate fraction of conductive layer that is porous ice "
MC.f_phi = (np.log(MC.T_phi/MC.T_s))/(np.log(MC.T_cond_base/MC.T_s)) # Porous fraction


" Calculate contribution of porosity to thermal conductivity "
# Thermal conductivity with only T dependence, ignoring salt and porosity
T_T  = (MC.T_s+MC.T_cond_base)/2 # Average T over conductive layer [MC.K]
MC.k_T  = 567/T_T # Average k(T) [W/m MC.K]

# Thermal conductivity in the porous region of the conductive layer
T_pore  = (MC.T_s+MC.T_phi)/2 # Average T over porous layer [MC.K]
k_pore  = 567/T_pore # Average k_porous(T,MC.phi) [W/m MC.K]

# Thermal conductivity in the solid region of the conductive layer
T_solid = (MC.T_phi+MC.T_cond_base)/2 # Average T over non-porous layer [MC.K]
k_solid = 567/T_solid # Average k_nonporous(T,MC.phi) [W/m MC.K]

# Weighted mean conductivity due to temperature and porosity
MC.k_Tp = (MC.f_phi * k_pore * (1-MC.phi)) + ((1-MC.f_phi) * k_solid) # MC.K(T,MC.phi) [W/m MC.K]


" H2O Layer Characteristic Densities "
# Begin with potential constiuent salts
rho_NaCl = 2160; # NaCl density [kg/m^3]
rho0gSO4 = 2660; # MgSO4 density [kg/m^3]
rho_natron = 1440; # Na2CO3•10(H2O) density [kg/m^3]

# Non-ice density
rho_s_mean = np.mean([rho_NaCl, rho0gSO4, rho_natron]) # Mean salt density [kg/m^3]
rho_s_un = np.std([rho_NaCl, rho0gSO4, rho_natron]) # Uncertainty
rho_s_sig = 1 # M sigma uncertainty 
# Sample normal distribution
MC.rho_s = np.random.normal(rho_s_mean, rho_s_un/rho_s_sig, MC.N) # Salt density [kg/m^3]

# Densities of conductive sublayers as a function of temperature
rho_T_pore = (1 + alpha * (MC.T_m - T_pore)) * rho0 # Density in porous layer [kg/m^3]
rho_T_solid =  (1 + alpha * (MC.T_m - T_solid)) * rho0 # Density in non-porous cond. layer [kg/m^3]

# Incorporate non-ice composition in the conductive sublayer
rho_T_s_pore = rho_T_pore * (1-MC.f_s)  + MC.f_s * MC.rho_s # Density in porous layer [kg/m^3]
rho_T_solid = rho_T_solid * (1-MC.f_s)  + MC.f_s * MC.rho_s # Density in non-porous cond. layer [kg/m^3]

# Incorporate porosity and find characteristic conductive layer density
MC.rho_cond =  (MC.f_phi * rho_T_s_pore * (1-MC.phi)) + ((1-MC.f_phi) * rho_T_solid) # Conductive density [kg/m^3]

# Find convective layer density for later reporting
MC.rho_conv =  (1 + alpha * (MC.T_m - MC.T_c)) * rho0 # Convective density [kg/m^3]

# Find ocean density for later reporting
MC.rho_ocn = 1000 * (1-MC.f_s)  + MC.f_s * MC.rho_s # Ocean density [kg/m^3]


" Calculate contribution of non-ice materials to thermal properties "
MC.B_k = 10**np.random.uniform(-1,1,MC.N) # Salt thermal conductivity scaling coefficient
k_s = MC.k_Tp * MC.B_k # Thermal conductivity of non-ice materials [W/m MC.K]
  
# Full characteristic thermal conductivity of conductive layer
MC.k_Tps = ((1-MC.f_s) * MC.k_Tp) + (MC.f_s * k_s) #  k(T,MC.phi,fs) [W/m MC.K]

# Full characteristic thermal diffusivity f conductive layer 
MC.K = MC.k_Tps / (MC.rho_cond * Cp) # MC.K(T,MC.phi,fs) [m^2/s]







"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
STEP 3: SAMPLE THE CONDUCTIVE THICKNESS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
print('Sampling Conductive Thickness')


" Calculate the melting temperature viscosity of ice "
# Volume diffusion coefficient
Dv = MC.D0v*np.exp(-MC.Qv/(R*MC.T_c))
# Boundary diffusion coefficient
Db = MC.D0b*np.exp(-MC.Qb/(R*MC.T_c))

# Grain size 
d_mean = np.log10(10**(-3)) # Grain size order mean [log10(m)]
d_un = 1 # Uncertainty in orders of magnitude
d_sig = 1 # M sigma uncertainty 
# Sample normal distribution
MC.d = 10**np.random.normal(d_mean, d_un/d_sig, MC.N) # Grain size [m]

# Grain boundary width input measurements
d_del_0 = 9.04e-10 # Grain boundary width for ice [m] (Frost and Ashby, 1982)
d_del_1 = 5.22e-10 # Grain boundary width as 2x Burgers Vector of Hondoh 2019 [m]

# Grain boundary width distribution
d_del_mean = np.mean([d_del_0, d_del_1]) # Mean grain boundary width [m]
d_del_un = np.std([d_del_0, d_del_1]) # Uncertainty [m]
d_del_sig = 1 # M sigma uncertainty 
# Sample normal distribution
MC.d_del = np.random.normal(d_del_mean, d_del_un/d_del_sig, MC.N)# Grain boundary width [m]

# Viscoisty of the convecting ice
MC.eta_c = (1/2)*((42 * d_Vm / (R * MC.T_c * MC.d**2)) * (Dv + (np.pi*MC.d_del/MC.d)*Db))**-1 # Convective viscosity [Pa s]

# Watch for non-physical extreme outliers
if np.any(MC.eta_c<0):
    MC = EMC.remove(MC,MC.eta_c<0)
MC = EMC.remove(MC,[np.log10(MC.eta_c)<10])
MC = EMC.remove(MC,[np.log10(MC.eta_c)>25])


" Bound Conductive Thickness "


# Elastic Shear Modulus 
G_mean = 3.5e9 # Elastic shear modulus [Pa]
G_un = 0.5e9 # Uncertainty
G_sig = 1 # M sigma uncertainty 
# Sample truncated normal distribution
MC.G_conv = G_mean*truncnorm.rvs(a=-np.inf, b=0, loc = 1, scale = (G_un/G_sig)/G_mean, size=MC.N)
MC = EMC.remove(MC,[MC.G_conv<G_mean/20])
# Conductive layer rigitity
MC.G_cond = (MC.f_phi * MC.G_conv * (1-MC.phi)) + ((1-MC.f_phi) * MC.G_conv) # Conductive layer rigitity [Pa]

# Tidal strain amplitude 
epsilon_mean = 1e-5 # Tidal flexing  mean (Showman and Han, 2004)
epsilon_un = 0.05*1e-5 # Uncertainty
epsilon_sig = 1# M sigma uncertainty 
# Sample normal distribution
MC.epsilon = np.random.normal(epsilon_mean, epsilon_un/epsilon_sig, MC.N) # Tidal strain amplitude [1/s]
# Get convective heat production per unit layer depth [W/m^3]
MC.dqc_dz = ((MC.epsilon)**2 * omega**2 * MC.eta_c / (1+omega**2 * MC.eta_c**2 / MC.G_conv**2))

# Estimate minimum conductive layer heat production
D_conv_min = (567/MC.T_c) * (MC.T_m - MC.T_cond_base) / MC.q_i
q_conv_min = MC.dqc_dz * D_conv_min

# Critical Rayleigh number
Ra_crit = 1e6 # McKinnon 1999; Assuming rheological uncertainty is absorbed in calculation of viscosity
# Find maximum conductive thickness before convection occurs
D_max_Ra = ((Ra_crit * MC.K * MC.eta_c)/(alpha * (MC.T_cond_base - MC.T_s) * rho0 * g))**(1/3) # Ra-bounded max cond. thickness [m]
# Find maximum conductive thickness 
D_max_q_i = MC.k_Tps * (MC.T_cond_base - MC.T_s) / (MC.q_i+q_conv_min) # Heat flux bounded max cond. thickness [m]
# Find smallest value bounding maximum conductive thickness
D_max_Ra[np.isnan(D_max_Ra)] = np.inf # Ra-bounded thickness physical limit check
MC.D_max = np.minimum(D_max_q_i, D_max_Ra) # Final max. cond. thickness.

# Get maximum convective dissipation [W/m^2]
q_c_max = MC.D_H2O * MC.dqc_dz

# Calculate minimum conductive thickness from maximum convetive dissipation
MC.D_min = MC.k_Tps * (MC.T_cond_base - MC.T_s) / (q_c_max+MC.q_i) # Minimum conductive thickness [m]







"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
STEP 4: DETERMINE LAYER AND TOTAL THICKNESSES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
print('Calculating Convective Thickness')


" Sample Conductive Thickness "
# Sample conductive thickness from otherwise uninformed prior 
MC.D_cond = np.random.uniform(MC.D_min,MC.D_max,MC.N) # Conductive thickness [m]
# Approximate brittle thickness for example comparison
MC.D_brittle = MC.D_cond*MC.f_phi


" Calculate convective thickness required to maintain conductive thickness selection "
# Required convective heat flux [W/m^2]
MC.q_c    = MC.k_Tps * (MC.T_cond_base - MC.T_s) / MC.D_cond - MC.q_i 
# Surface heat flux [W/m^2]
MC.q_s = MC.q_i + MC.q_c
# Convective thickness [m]
MC.D_conv = MC.q_c / MC.dqc_dz


" Calculate total thickness "
MC.D_tot = MC.D_cond + MC.D_conv # Total Thickness [W/m^2]

# Remove remaining non-physical outliers
MC = EMC.remove(MC,[MC.D_bnd > MC.D_conv])
MC = EMC.remove(MC,[MC.D_min > MC.D_max])
MC = EMC.remove(MC,[MC.D_cond > MC.D_H2O])
MC = EMC.remove(MC,[MC.D_conv > MC.D_H2O])
MC = EMC.remove(MC,[MC.D_tot > MC.D_H2O])
MC = EMC.remove(MC,[MC.D_cond > MC.D_max])







"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ANALYSIS: CALCULATE OBSERVABLES FOR COMPARISON
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""  
# Invert for k2/Q
MC.k2Q_out = MC.q_s/(((21/2) * n_Europa**5 * r_E**5 * e_Europa**2 / UGC) / SA_Europa)

" Finish interior structure "
# Iron Core Radius (Tobie 2003; O.L.Kuskov & V.A.Kronrod 2005)
D_iron_mean = np.mean([455e3,670e3,555,660,505,630,470,610,455,670,560,670,510,640,490,620,470,670]) # Mean iron core radius [m], O.L.Kuskov & V.A.Kronrod 2005
D_iron_un = np.std([455e3,670e3,555,660,505,630,470,610,455,670,560,670,510,640,490,620,470,670]) # Uncertainty
D_iron_sig = 3 # M sigma uncertainty 
# Sample normal distribution
MC.D_iron = np.random.normal(D_iron_mean, D_iron_un/D_iron_sig, MC.N) # Iron core radius [m]

# Potential densitues of iron core [kg/m^3]
rho_iron_lowS = 5700; 
rho_iron_hiS = 4700; 

# Iron core density
rho_iron_mean = np.mean([rho_iron_lowS, rho_iron_hiS]) # Mean core density [kg/m^3]
rho_iron_un = np.std([rho_iron_lowS, rho_iron_hiS]) # Uncertainty
rho_iron_sig = 3 # M sigma uncertainty 
# Sample normal distribution
MC.rho_iron = np.random.normal(rho_iron_mean, rho_iron_un/rho_iron_sig, MC.N) # Iron core density [kg/m^3]

# Silicate thickness
MC.D_rock = r_E - MC.D_H2O - MC.D_iron # Silicate thickness [m]

# Build layers
r0 = r_E # Top of cond layer
r1 = r_E - MC.D_cond # Bottom of cond layer
r2 = r1 - MC.D_conv #Bottom of convective layer
r3 = r_E - MC.D_H2O # Seafloor
r4 = MC.D_iron # Top of iron core

# Silicate density [kg/m^3]
MC.rho_rock = (m_E -
           (4/3)*np.pi*(r0**3 - r1**3)*MC.rho_cond - \
           (4/3)*np.pi*(r1**3 - r2**3)*MC.rho_conv - \
           (4/3)*np.pi*(r2**3 - r3**3)*MC.rho_ocn - \
           (4/3)*np.pi*(r4**3)*MC.rho_iron)/((4/3)*np.pi*(r3**3-r4**3))

# Get contributions to MC.MoI of layers
C_MoI_cond = (8/15)*np.pi*MC.rho_cond*(r0**5 - r1**5) # Conductive ice moment of inertia
C_MoI_conv = (8/15)*np.pi*MC.rho_conv*(r1**5 - r2**5) # Convective ice moment of inertia
C_MoI_ocn =  (8/15)*np.pi*MC.rho_ocn*(r2**5 - r3**5) # Ocean moment of inertia
C_MoI_rock = (8/15)*np.pi*MC.rho_rock*(r3**5 - r4**5) # Silicate moment of inertia
C_MoI_core = (8/15)*np.pi*MC.rho_iron*r4**5 # Core moment of inertia

# Polar Moment of inertia
C_MoI = C_MoI_cond + C_MoI_conv + C_MoI_ocn + C_MoI_rock + C_MoI_core

# Polar moment of inertia factor
MC.MoI = C_MoI / (m_E * r_E**2) 







"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ANALYSIS: CALCULATE CBE VALUES AND UNCERTAINTIES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""  

# Here, all distributions are calculated for P(y | D_tot) based on the 
# full pdf of D_tot, noted by *_cond

# Roll the dice! for conditional filtering
dice = np.random.rand(MC.N)/2 + 0.5

" Layer Thicknesses "
# Total Thickness 
D_tot_CBE = EMC.CBEanalysis(MC.D_tot)
# Conductive Thickness 
D_cond_cond = EMC.condAnalysis(MC.D_cond,MC.D_tot,D_tot_CBE.CBE,dice)
D_cond_CBE = EMC.CBEanalysis(D_cond_cond)
# Convective Thickness #
D_conv_cond = EMC.condAnalysis(MC.D_conv,MC.D_tot,D_tot_CBE.CBE,dice)
D_conv_CBE = EMC.CBEanalysis(D_conv_cond)
# Porous fraction of conductive layer (~brittle)
f_phi_cond = EMC.condAnalysis(MC.f_phi,MC.D_tot,D_tot_CBE.CBE,dice)
f_phi_CBE = EMC.CBEanalysis(f_phi_cond)
# Porous thickness
D_brittle_cond = EMC.condAnalysis(MC.D_brittle,MC.D_tot,D_tot_CBE.CBE,dice)
D_brittle_CBE = EMC.CBEanalysis(D_brittle_cond)


" Heat Fluxes "
# Interior Heat Flux 
q_i_cond = EMC.condAnalysis(MC.q_i,MC.D_tot,D_tot_CBE.CBE,dice)
q_i_CBE = EMC.CBEanalysisHeatflow(q_i_cond)
# Radiative Heat Flux
q_ir_cond = EMC.condAnalysis(MC.q_ir,MC.D_tot,D_tot_CBE.CBE,dice)
q_ir_CBE = EMC.CBEanalysisHeatflow(q_ir_cond)
# Rocky Tidal Heat Flux
q_it_cond = EMC.condAnalysis(MC.q_it,MC.D_tot,D_tot_CBE.CBE,dice)
q_it_CBE = EMC.CBEanalysisHeatflow(q_it_cond)
# Convective Heat Flux
q_c_cond = EMC.condAnalysis(MC.q_c,MC.D_tot,D_tot_CBE.CBE,dice)
q_c_CBE = EMC.CBEanalysisHeatflow(q_c_cond)
# Surface Heat Flux
q_s_cond = EMC.condAnalysis(MC.q_s,MC.D_tot,D_tot_CBE.CBE,dice)
q_s_CBE = EMC.CBEanalysisHeatflow(q_s_cond)


" Temperatures "
# Conductive Temperature 
T_cond_base_cond = EMC.condAnalysis(MC.T_cond_base,MC.D_tot,D_tot_CBE.CBE,dice)
T_cond_base_CBE = EMC.CBEanalysis(T_cond_base_cond)

T_c_cond = EMC.condAnalysis(MC.T_c,MC.D_tot,D_tot_CBE.CBE,dice)
T_c_CBE = EMC.CBEanalysis(T_c_cond)

" Rheology "
eta_c_cond = EMC.condAnalysis(MC.eta_c,MC.D_tot,D_tot_CBE.CBE,dice)
eta_c_CBE = EMC.CBEanalysis(np.log10(eta_c_cond))

" Thermal Properties "
# Thermal Conductivity k(T)
k_T_cond = EMC.condAnalysis(MC.k_T,MC.D_tot,D_tot_CBE.CBE,dice)
k_T_CBE = EMC.CBEanalysis(k_T_cond)
# Thermal Conductivity k(T,phi)
k_Tp_cond = EMC.condAnalysis(MC.k_Tp,MC.D_tot,D_tot_CBE.CBE,dice)
k_Tp_CBE = EMC.CBEanalysis(k_Tp_cond)
# Thermal Conductivity k(T,phi,f_s)
k_Tps_cond = EMC.condAnalysis(MC.k_Tps,MC.D_tot,D_tot_CBE.CBE,dice)
k_Tps_CBE = EMC.CBEanalysis(k_Tps_cond)

" Densities "
# Full conductive density
rho_cond_cond = EMC.condAnalysis(MC.rho_cond,MC.D_tot,D_tot_CBE.CBE,dice)
rho_cond_CBE = EMC.CBEanalysis(rho_cond_cond)
# Convective Density
rho_conv_cond = EMC.condAnalysis(MC.rho_conv,MC.D_tot,D_tot_CBE.CBE,dice)
rho_conv_CBE = EMC.CBEanalysis(rho_conv_cond)

" Measurables "
# MoI factor
MoI_cond = EMC.condAnalysis(MC.MoI,MC.D_tot,D_tot_CBE.CBE,dice)
MoI_CBE = EMC.CBEanalysis(MoI_cond)
# k2/Q
k2Q_cond = EMC.condAnalysis(MC.k2Q_out,MC.D_tot,D_tot_CBE.CBE,dice)
k2Q_CBE = EMC.CBEanalysis(k2Q_cond)







"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ANALYSIS: COMPARISON TO BILLINGS AND KATTENHORN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""  

" Billings and Kattenhorn full thickness"
# This is a pre-computed mean and stddev
D_full = np.random.normal(42.0e3, 46.6e3, MC.N)
BK_tot_CBE = EMC.CBEanalysis(D_full) # BK estimated full thickness 

# Must interpolate to the same bin
fD_tot = interpolate.interp1d(D_tot_CBE.bins,  D_tot_CBE.Psmooth, fill_value=(0,0),bounds_error=False)
D_tot_BK = fD_tot(BK_tot_CBE.bins) # Total MC thickness interpolated to BK range

# Convolve
P_BK_tot = D_tot_BK * BK_tot_CBE.Psmooth/np.max(D_tot_BK * BK_tot_CBE.Psmooth) # Fully convolved total thickness [m]


" Billings and Kattenhorn brittle thickness"
# This is a pre-computed mean and stddev
bk_brit_mean = 1.7e3 # Mean [km]
bk_brit_un = 2.1e3 # Uncertaintly
bk_brit_sig = 1 # M sigma uncertainty 
# Sample normal distribution
BK_brittle = np.random.normal(bk_brit_mean, bk_brit_un/bk_brit_sig, MC.N)
# Get distribution back
BK_brittle_CBE = EMC.CBEanalysis(BK_brittle) # BK estimated brittle thickness 

# Approximate brittle thickness from MC model
D_brittle = MC.D_cond*MC.f_phi # MC ~brittle thickness [m]
D_brittle_CBE = EMC.CBEanalysis(D_brittle) # MC ~brittle thickness 

# Determine if full thickness is likely given constraint
z_bk_brit = (D_brittle-bk_brit_mean)/bk_brit_un #zscore
P_bk_brit = st.norm.cdf(np.abs(z_bk_brit)) # percentile

# Roll the dice!
bk_brit_dice = np.random.rand(MC.N)/2 + 0.5
remove = np.zeros(MC.N)
remove[bk_brit_dice < P_bk_brit] = 1
# Filter out unlikely events
D_tot_BK_brittle = MC.D_tot[np.where(remove == 0)]

BK_tot_brittle_CBE = EMC.CBEanalysis(D_tot_BK_brittle) # BK estimated brittle thickness 


" Both Constraints -- full and brittle ice shell "
# Must interpolate to the same bin
fD_tot = interpolate.interp1d(BK_tot_brittle_CBE.bins,  BK_tot_brittle_CBE.Psmooth, fill_value=(0,0),bounds_error=False)
D_tot_BK = fD_tot(BK_tot_CBE.bins) # Total MC thickness interpolated to BK range

# Find CBE
bins = BK_tot_CBE.bins
P_BK_Full = (P_BK_tot * BK_tot_brittle_CBE.Psmooth)/np.max(P_BK_tot * BK_tot_brittle_CBE.Psmooth)
CBE_ind = np.where(P_BK_Full == max(P_BK_Full)) 
CBE = bins[CBE_ind]

# Cumulative probability
n_cum = np.cumsum(P_BK_Full)/np.max(np.cumsum(P_BK_Full));
CBE_cum = n_cum[CBE_ind]
CBE_cum_lo = CBE_cum * 0.341
CBE_cum_hi = (1-CBE_cum) * 0.341

CBE_lo = CBE - bins[(np.abs(n_cum - (CBE_cum-CBE_cum_lo))).argmin()]
CBE_hi = bins[(np.abs(n_cum - (CBE_cum+CBE_cum_hi))).argmin()] - CBE

# Store results - kg/m^3
D_BK_quantile = CBE_cum 
D_BK_CBE = EMC.MCCBE(CBE,CBE_hi,CBE_lo,bins,P_BK_Full,n_cum)









if not doStats:
    raise SystemExit(0)


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ANALYSIS: CORRELATIONS AND p-VALUES
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
print('Testing Relationships - May take an hour for N=1e7, a day for N=1e8')

varNum = 0;
varData = [MC.D_tot, MC.D_cond, MC.D_conv, MC.D_H2O, MC.D_rock, MC.D_iron, # 6 entries
           MC.T_s, MC.T_c, MC.T_phi, MC.T_m, MC.T_cond_base, # 5 entries
           MC.q_s,MC.q_i, MC.q_ir, MC.q_it, MC.q_c,  # 4 entries
           MC.rho_cond, MC.rho_conv, MC.rho_ocn, MC.rho_rock, MC.rho_iron, # 5 entries
           MC.G_conv, MC.G_cond, # 2 entries
           MC.d, MC.d_del, MC.eta_c, MC.D0v, MC.D0b, MC.Qv, MC.Qb, MC.epsilon,  # 8 entries
           MC.k2Q_out, MC.f_s, MC.B_k, MC.k_Tps, MC.phi, MC.f_phi]  # 7 entries

rowHeads =["D_tot", "D_cond", "D_conv", "D_H2O", "D_rock", "D_iron", 
           "T_s", "T_c", "T_phi", "T_m", "T_cond_base",
           "q_s","q_i", "q_ir", "q_it", "q_c",  
           "rho_cond", "rho_conv", "rho_ocn", "rho_rock", "rho_iron", 
           "G_conv", "G_cond", 
           "d", "d_del", "eta_c", "D0v", "D0b", "Qv", "Qb", "epsilon",  
           "k2Q_out", "f_s", "B_k", "k_Tps", "phi", "f_phi"]

colHeads = ['D_tot C','D_tot p','D_cond C','D_cond p','D_conv C','D_conv p']

CorrelationTable = np.zeros([len(varData),6])

for data in varData :
    
    CorrelationTable[varNum,0], CorrelationTable[varNum,1] = spearmanr(data, MC.D_tot)
    CorrelationTable[varNum,2], CorrelationTable[varNum,3] = spearmanr(data, MC.D_cond)
    CorrelationTable[varNum,4], CorrelationTable[varNum,5] = spearmanr(data, MC.D_conv)

    varNum = varNum + 1
    


pd.DataFrame(CorrelationTable).to_csv("Correlations.csv",header=colHeads,
                                      index=True)
pd.DataFrame(rowHeads).to_csv("Correlations.csv",mode='a',header=False)







if not doPlots:
    raise SystemExit(0)


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FIGURE 3: MINOR DISTRIBUTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""  
titleSize = 18
axisSize = 14
tickSize = 14

" Heat Flux "
fig = plt.figure(1); 
plt.clf()

# Plot
plt.title('Heat Flux', fontsize=titleSize)

plt.plot(q_i_CBE.bins,q_i_CBE.Psmooth*0.9)
plt.plot(q_ir_CBE.bins,q_ir_CBE.Psmooth*0.9)
plt.plot(q_it_CBE.bins,q_it_CBE.Psmooth*0.9)
plt.plot(q_c_CBE.bins,q_c_CBE.Psmooth*0.9)
plt.plot(q_s_CBE.bins,q_s_CBE.Psmooth*0.9)

plt.xlabel('log10(Heat Flux [W/m^2])', fontsize=axisSize)
plt.xlim([-3, 0])
plt.ylim([0, 1])
plt.legend(['Rock','Rad','RockTidal','IceTidal','Surface'])

fig.savefig("q_all.pdf", bbox_inches='tight')


" Thermal Conductivity "
fig = plt.figure(2); 
plt.clf()

# Plot
plt.title('Thermal Conductivity', fontsize=titleSize)

plt.plot(k_T_CBE.bins,k_T_CBE.Psmooth*0.9)
plt.plot(k_Tp_CBE.bins,k_Tp_CBE.Psmooth*0.9)
plt.plot(k_Tps_CBE.bins,k_Tps_CBE.Psmooth*0.9)


plt.xlabel('Thermal Conductivity [W/m K]', fontsize=axisSize)
plt.xlim([2, 6])
plt.ylim([0, 1])
plt.legend(['k(T)','k(T,phi)','k(T,phi,f_s)'])

fig.savefig("k_all.pdf", bbox_inches='tight')


" Viscosity "
fig = plt.figure(3); 
plt.clf()

# Plot
plt.title('Viscosity', fontsize=titleSize)

plt.plot(eta_c_CBE.bins,eta_c_CBE.Psmooth*0.9)


plt.xlabel('log10(Viscosity [Pa s])', fontsize=axisSize)
plt.xlim([10, 22])
plt.ylim([0, 1])

fig.savefig("eta_c.pdf", bbox_inches='tight')







"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FIGURE 4: TOTAL AND LAYER THICKNESSES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""  

" Total Thickness "
fig = plt.figure(4); 
plt.clf()
ax1 = plt.subplot(111)

# Get quantiles
MCquantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.90]
MCquantiles = np.sort(np.append(MCquantiles, stats.percentileofscore(MC.D_tot,D_tot_CBE.CBE)/1e2))
xTicks = np.quantile(MC.D_tot, MCquantiles)/1e3
xLabels = ["%.2f" % number for number in MCquantiles]

# Plot
plt.title('Total Thickness', fontsize=titleSize)
ax2 = ax1.twiny()
plt.ylim([0, 1])

ax1.plot(D_tot_CBE.bins/1e3,D_tot_CBE.Psmooth*0.9)
ax1.plot(D_tot_CBE.bins/1e3,D_tot_CBE.Pcum)

ax1.set_xlabel('Thickness [km]', fontsize=axisSize)
ax1.set_xlim([0, 160])

ax2.set_xlabel('Quantile', fontsize=axisSize)
ax2.set_xlim([0, 160])
ax2.set_xticks(xTicks)
ax2.set_xticklabels(xLabels)

fig.savefig("d_tot.pdf", bbox_inches='tight')


" Layer Thicknesses "
fig = plt.figure(5); 
plt.clf()

# Plot
plt.title('Layer Thicknesses', fontsize=titleSize)

plt.plot(D_cond_CBE.bins/1e3,D_cond_CBE.Psmooth*0.9)
plt.plot(D_conv_CBE.bins/1e3,D_conv_CBE.Psmooth*0.9)

plt.xlabel('Thickness [km]', fontsize=axisSize)
plt.xlim([0, 60])
plt.ylim([0, 1])

fig.savefig("d_layers.pdf", bbox_inches='tight')







"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FIGURE 5: HEATMAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""  

" Full Heatmap "
fig = plt.figure(7); 
plt.clf()

# Get quantiles
ax = fig.add_subplot(1, 1, 1)
plt.hist2d(MC.D_cond/1e3,MC.D_conv/1e3,bins=1000,range=[[0,60],[0,160]],cmap="inferno",norm=mpl.colors.LogNorm())

ax.set_facecolor('black')
plt.gca().set_aspect('equal', adjustable='box')

# I reccomend a screen cap--trying to vectorize heat maps may freeze your machine
# fig.savefig("full_heatmap.pdf", bbox_inches='tight')
fig.savefig("full_heatmap.png", bbox_inches='tight')

" Zoomed full heatmap "
fig = plt.figure(8); 
plt.clf()

# Get quantiles
ax = fig.add_subplot(1, 1, 1)
plt.hist2d(MC.D_cond/1e3,MC.D_conv/1e3,bins=300,range=[[0,30],[0,30]],cmap="inferno",norm=mpl.colors.LogNorm())

ax.set_facecolor('black')
plt.gca().set_aspect('equal', adjustable='box')

# I reccommend a png--trying to vectorize heat maps may freeze your machine
fig.savefig("full_heatmap_zoomed.png", bbox_inches='tight')



" Conductive  heatmap "
fig = plt.figure(9); 
plt.clf()

# Get quantiles
ax = fig.add_subplot(1, 1, 1)
plt.hist2d(MC.D_tot/1e3,MC.D_cond/1e3,bins=1000,range=[[0,160],[0,160]],cmap="inferno",norm=mpl.colors.LogNorm())

ax.set_facecolor('black')
plt.gca().set_aspect('equal', adjustable='box')

# I reccommend a png--trying to vectorize heat maps may freeze your machine
# fig.savefig("cond_heatmap.pdf", bbox_inches='tight')
fig.savefig("cond_heatmap.png", bbox_inches='tight')


" Convective  heatmap "
fig = plt.figure(10); 
plt.clf()

# Get quantiles
ax = fig.add_subplot(1, 1, 1)
plt.hist2d(MC.D_tot/1e3,MC.D_conv/1e3,bins=1000,range=[[0,160],[0,160]],cmap="inferno",norm=mpl.colors.LogNorm())

ax.set_facecolor('black')
plt.gca().set_aspect('equal', adjustable='box')

# I reccomend a screen cap--trying to vectorize heat maps may freeze your machine
# fig.savefig("cond_heatmap.pdf", bbox_inches='tight')
fig.savefig("conv_heatmap.png", bbox_inches='tight')







"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FIGURE 6: BK CONSTRAINED LAYER THICKNESSES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""  

" Constraints "
fig = plt.figure(11); 
plt.clf()

# Plot
plt.title('Constraints', fontsize=titleSize)

plt.plot(D_tot_CBE.bins/1e3,D_tot_CBE.Psmooth*0.9)
plt.plot(BK_tot_CBE.bins/1e3,P_BK_tot*0.9)
plt.plot(BK_tot_brittle_CBE.bins/1e3,BK_tot_brittle_CBE.Psmooth*0.9)
plt.plot(D_BK_CBE.bins/1e3,D_BK_CBE.Psmooth*0.9)

plt.legend(['MC','Total Const.','Brittle Const.','Both Const.'])
plt.xlabel('Thickness [km]', fontsize=axisSize)
plt.xlim([0, 160])
plt.ylim([0, 1])

fig.savefig("d_BK_constraints.pdf", bbox_inches='tight')



" Full BK Thickness  "
fig = plt.figure(12); 
plt.clf()
ax1 = plt.subplot(111)

# Plot
plt.title('Full BK Thickness', fontsize=titleSize)

# Get quantiles
MCquantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.90]
MCquantiles = np.sort(np.append(MCquantiles, D_BK_quantile))
xTicks = np.zeros(np.size(MCquantiles))

for i in range(np.size(MCquantiles)):
    xTicks[i] = np.max(D_BK_CBE.bins[D_BK_CBE.Pcum<=MCquantiles[i]])/1e3

xLabels = ["%.2f" % number for number in MCquantiles]

# Plot
plt.title('Total Thickness', fontsize=titleSize)
ax2 = ax1.twiny()
plt.ylim([0, 1])

ax1.plot(D_BK_CBE.bins/1e3,D_BK_CBE.Psmooth*0.9)
ax1.plot(D_BK_CBE.bins/1e3,D_BK_CBE.Pcum)

ax1.set_xlabel('Thickness [km]', fontsize=axisSize)
ax1.set_xlim([0, 160])

ax2.set_xlabel('Quantile', fontsize=axisSize)
ax2.set_xlim([0, 160])
ax2.set_xticks(xTicks)
ax2.set_xticklabels(xLabels)

fig.savefig("d_BK_full.pdf", bbox_inches='tight')












