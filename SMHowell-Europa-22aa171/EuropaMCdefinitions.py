#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt 
from scipy.signal import savgol_filter
from scipy import interpolate

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

class MCarrays:
    "This class contains the important arrays"
    def __init__(self, N):  
        # Layers
        self.D_tot = np.zeros(N)
        self.D_cond = np.zeros(N)
        self.D_min = np.zeros(N)
        self.D_max = np.zeros(N)
        self.D_brittle = np.zeros(N)
        self.D_conv = np.zeros(N)
        self.D_bnd = np.zeros(N)
        self.D_H2O = np.zeros(N)
        self.D_ocn = np.zeros(N)
        self.D_rock = np.zeros(N)
        self.D_iron = np.zeros(N)
        
        # Temperatures
        self.T_s = np.zeros(N)
        self.T_c = np.zeros(N)
        self.T_phi = np.zeros(N) 
        self.T_m = np.zeros(N) 
        self.T_cond_base = np.zeros(N)
        
        # Heat fluxes
        self.q_i = np.zeros(N)
        self.q_ir = np.zeros(N)
        self.q_it = np.zeros(N)
        self.q_c = np.zeros(N)
        self.q_s = np.zeros(N)
        self.dqc_dz = np.zeros(N)
        self.Q_ir = np.zeros(N)  
        self.Pi = np.zeros(N)  
        
        # Densities
        self.rho_cond = np.zeros(N)
        self.rho_conv = np.zeros(N)
        self.rho_ocn = np.zeros(N)
        self.rho_rock = np.zeros(N)
        self.rho_iron = np.zeros(N)      
        self.rho_s = np.zeros(N)    
        
        # Elastic Parameters
        self.G_conv = np.zeros(N)
        self.G_cond = np.zeros(N) 
        
        # Grain geometry
        self.d = np.zeros(N)
        self.d_del = np.zeros(N)
        
        # Rheology
        self.eta_c = np.zeros(N)
        self.D0v = np.zeros(N)
        self.D0b = np.zeros(N)
        self.Qv = np.zeros(N)
        self.Qb = np.zeros(N)
        self.epsilon = np.zeros(N) 
        
        # Measureables
        self.MoI = np.zeros(N)   
        self.k2Q_in = np.zeros(N)   
        self.k2Q_out = np.zeros(N)   
        self.k2Q_lambda = np.zeros(N)   
        
        # Thermal/mechanical  properties
        self.f_s = np.zeros(N)
        self.B_k = np.zeros(N)
        self.k_T = np.zeros(N)
        self.k_Tp = np.zeros(N)
        self.k_Tps = np.zeros(N)
        self.K = np.zeros(N)
        self.phi = np.zeros(N)      
        self.f_phi = np.zeros(N)    
             
        # Update N
        self.N = N
        

class MCCBE:
    def __init__(self, CBE, hi, lo, bins, Psmooth, Pcum): 
        self.CBE = CBE
        self.hi = hi
        self.lo = lo
        self.bins = bins
        self.Psmooth = Psmooth
        self.Pcum = Pcum
        
def remove(MC,criteria):
    badValues = np.zeros(MC.N)
    badValues[tuple(criteria)] = 1
        
    # Layers
    MC.D_tot  = MC.D_tot[np.where(badValues == 0)]
    MC.D_cond = MC.D_cond[np.where(badValues == 0)]
    MC.D_min = MC.D_min[np.where(badValues == 0)]
    MC.D_max = MC.D_max[np.where(badValues == 0)]
    MC.D_brittle = MC.D_brittle[np.where(badValues == 0)]
    MC.D_conv = MC.D_conv[np.where(badValues == 0)]
    MC.D_bnd = MC.D_bnd[np.where(badValues == 0)]
    MC.D_H2O = MC.D_H2O[np.where(badValues == 0)]
    MC.D_ocn = MC.D_H2O - MC.D_tot
    MC.D_rock = MC.D_rock[np.where(badValues == 0)]
    MC.D_iron = MC.D_iron[np.where(badValues == 0)]
    
    # Temperatures
    MC.T_s = MC.T_s[np.where(badValues == 0)]
    MC.T_c = MC.T_c[np.where(badValues == 0)]
    MC.T_phi = MC.T_phi[np.where(badValues == 0)]
    MC.T_m = MC.T_m[np.where(badValues == 0)]
    MC.T_cond_base = MC.T_cond_base[np.where(badValues == 0)]
    
    # Heat fluxes
    MC.q_i = MC.q_i[np.where(badValues == 0)]
    MC.q_ir = MC.q_ir[np.where(badValues == 0)]
    MC.q_it = MC.q_it[np.where(badValues == 0)]
    MC.q_c = MC.q_c[np.where(badValues == 0)]
    MC.q_s = MC.q_s[np.where(badValues == 0)]
    MC.dqc_dz = MC.dqc_dz[np.where(badValues == 0)]
    MC.Q_ir = MC.Q_ir[np.where(badValues == 0)]
    MC.Pi = MC.Pi[np.where(badValues == 0)]
    
    # Densities
    MC.rho_cond = MC.rho_cond[np.where(badValues == 0)]
    MC.rho_conv = MC.rho_conv[np.where(badValues == 0)]
    MC.rho_ocn = MC.rho_ocn[np.where(badValues == 0)]
    MC.rho_rock = MC.rho_rock[np.where(badValues == 0)]
    MC.rho_iron = MC.rho_iron[np.where(badValues == 0)]
    MC.rho_s = MC.rho_s[np.where(badValues == 0)]
    
    # Elastic Parameters
    MC.G_conv = MC.G_conv[np.where(badValues == 0)]
    MC.G_cond = MC.G_cond[np.where(badValues == 0)]
    
    # Grain geometry
    MC.d = MC.d[np.where(badValues == 0)]
    MC.d_del = MC.d_del[np.where(badValues == 0)]
    
    # Rheology
    MC.eta_c = MC.eta_c[np.where(badValues == 0)]
    MC.D0v = MC.D0v[np.where(badValues == 0)]
    MC.D0b = MC.D0b[np.where(badValues == 0)]
    MC.Qv = MC.Qv[np.where(badValues == 0)]
    MC.Qb = MC.Qb[np.where(badValues == 0)]
    MC.epsilon = MC.epsilon[np.where(badValues == 0)]
    
    # Measureables
    MC.MoI = MC.MoI[np.where(badValues == 0)]
    MC.k2Q_in = MC.k2Q_in[np.where(badValues == 0)]
    MC.k2Q_out = MC.k2Q_out[np.where(badValues == 0)]
    MC.k2Q_lambda = MC.k2Q_lambda[np.where(badValues == 0)]
    
    # Thermal/mechanical  properties
    MC.f_s = MC.f_s[np.where(badValues == 0)]
    MC.B_k = MC.B_k[np.where(badValues == 0)]
    MC.k_T = MC.k_T[np.where(badValues == 0)]
    MC.k_Tp = MC.k_Tp[np.where(badValues == 0)]
    MC.k_Tps = MC.k_Tps[np.where(badValues == 0)]
    MC.K = MC.K[np.where(badValues == 0)]
    MC.phi = MC.phi[np.where(badValues == 0)]
    MC.f_phi = MC.f_phi[np.where(badValues == 0)]  
    
    # Update N
    MC.N = np.size(MC.D_tot)
    
    return MC
    

    
    
def CBEanalysis(data):
    # Differential probability
    n, bins = np.histogram(data, bins=1000, density=True) # Get histogram
    n_smooth = savgol_filter(n, 51, 3)# Smooth a bit with Savitzky Golay
    n_smooth[n_smooth<0]=0 # To counter any unintended behavior at the ends
    CBE_ind = np.where(n_smooth == max(n_smooth)) 
    CBE = bins[CBE_ind]
    
    # Cumulative probability
    plt.figure(99)
    n_cum, bins_cum, patches = plt.hist(data, bins=1000, density=True, histtype='step', cumulative=True, label='Empirical')
    plt.close(99)
    CBE_cum = n_cum[CBE_ind]
    CBE_cum_lo = CBE_cum * 0.341
    CBE_cum_hi = (1-CBE_cum) * 0.341
    
    CBE_lo = CBE - bins[(np.abs(n_cum - (CBE_cum-CBE_cum_lo))).argmin()]
    CBE_hi = bins[(np.abs(n_cum - (CBE_cum+CBE_cum_hi))).argmin()] - CBE
        
    # Return data
    varOut = MCCBE(CBE,CBE_hi,CBE_lo,bins[0:-1],n_smooth/np.max(n_smooth),n_cum)
    return varOut

    


def CBEanalysisHeatflow(data):
    # Differential probability
    log_data = np.log10(data)
    n, bins = np.histogram(log_data, bins=1000, density=True) # Get histogram
    n_smooth = savgol_filter(n, 51, 3)# Smooth a bit with Savitzky Golay
    n_smooth[n_smooth<0]=0 # To counter any unintended behavior at the ends
    CBE_ind = np.where(n_smooth == max(n_smooth)) 
    CBE = bins[CBE_ind]
    
    # Cumulative probability
    plt.figure(99)
    n_cum, bins_cum, patches = plt.hist(data, bins=1000, density=True, histtype='step', cumulative=True, label='Empirical')
    plt.close(99)
    CBE_cum = n_cum[CBE_ind]
    CBE_cum_lo = CBE_cum * 0.341
    CBE_cum_hi = (1-CBE_cum) * 0.341
    
    CBE_lo = CBE - bins[(np.abs(n_cum - (CBE_cum-CBE_cum_lo))).argmin()]
    CBE_hi = bins[(np.abs(n_cum - (CBE_cum+CBE_cum_hi))).argmin()] - CBE
        
    CBE_hi = 10**(CBE+CBE_hi)-10**CBE
    CBE_lo = 10**(CBE-CBE_lo)
    CBE = 10**CBE
    
    # Return data
    varOut = MCCBE(CBE,CBE_hi,CBE_lo,bins[0:-1],n_smooth/np.max(n_smooth),n_cum)
    return varOut

def condAnalysis(data,thicknessSamples,thicknessCBE,dice):
    
    N = thicknessSamples.size
    lo_D = thicknessSamples[thicknessSamples<=thicknessCBE]
    hi_D = thicknessSamples[thicknessSamples>thicknessCBE]
    lo_D_CBE = CBEanalysis(lo_D)
    hi_D_CBE = CBEanalysis(hi_D)
    
    # Build a filter that with probability of rejection based on CBE thickness and 
    # distribution. 100% Probability of keeping CBE value, decays based on computed 
    # variance towards tails.
    fP_keep = interpolate.interp1d(np.append(lo_D_CBE.bins, hi_D_CBE.bins),np.append(lo_D_CBE.Pcum, 1-hi_D_CBE.Pcum), fill_value=(0,0),bounds_error=False)
    P_keep = fP_keep(thicknessSamples)
    remove = np.zeros(N)
    remove[dice > P_keep] = 1

    data = data[np.where(remove == 0)]
    return data





def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def getQuantiles(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    