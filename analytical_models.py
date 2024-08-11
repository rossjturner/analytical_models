# analytical_models module
# Ross Turner, 22 Mar 2023

# import packages
import numpy as np
import pandas as pd
import os, warnings
from astropy import constants as const

## Define global variables that can be adjusted to customise model output
# basic constants
year = 365.2422*24*3600 # average year in seconds
maverage = (0.6*const.m_p.value) # kg average particle mass

c_speed = const.c.value # speed of light
e_charge = const.e.value # electron charge
k_B = const.k_B.value # Boltzmann constant
m_e = const.m_e.value # electron mass
mu0 = const.mu0.value # vacuum permeability
sigma_T = const.sigma_T.value # electron scattering cross-section

# model parameters that can be optimised for efficiency
limTime = (year) # the FR-II limit must be used before this time
stepRatio = 1.01 # ratio to increase time/radius

# shocked gas and lobe parameters
chi = 2*np.pi/3.0 # lobe geometry parameter
equipartition = 0.03 # equipartition factor
gammaC = (5./3) # adiabatic index of lobe plasma
gammaJ = (4./3) # adiabatic index of jet plasma
gammaS = (5./3) # adiabatic index of shocked shell gas
gammaX = (5./3) # adiabatic index of external gas

# density and temperature profiles
rCutoff = 0.01 # minimum radius to match profiles as a fraction of r200
betaMax = 2 # set critical value above which the cocoon expands balistically


## Define main function to run dynamical models
def dynamical_run(jet_power, source_ages, opening_angle, betas, regions, rho0Value, temperature, model='scheuer'):
    """
    Function to calculate size, volume and pressure evolutionary histories for a given dynamical model.
        
    Parameters
    ----------
    jet_power : float
        double-sided jet kinetic power; in units of log Watts
    source_ages : float or list
        time step(s) to output dynamical quantities; in units of log years
    opening_angle : float
        half-opening angle of the jet; in units of degrees
        
    etc.
    """
    
    # create folder for output files if not present
    if not os.path.exists('LDtracks'):
        os.mkdir('LDtracks')
    
    ## AMBIENT MEDIUM
    # set maximum number of regions
    nregions = len(betas)

    # find values of density parameter in each beta region
    k0Value = rho0Value*regions[0]**betas[0]
    kValues = __DensityParameter(nregions, k0Value, betas, regions)
    
    # extend first beta region to a radius of zero
    new_regions = regions.copy()
    new_regions[0] = 0.
 
    # calculate dynamical evolution of lobe and shocked shell using RAiSE dynamics
    lengths, volumes, pressures = __runge_kutta(10**jet_power/2, 10**np.array(source_ages)*year, opening_angle, nregions, betas, regions, kValues, temperature, model)
    
    # create pandas dataframe for integrated emission
    df = pd.DataFrame()
    df['Time (yrs)'] = 10**np.asarray(source_ages).astype(np.float_)
    df['Size (kpc)'] = 2*lengths[:]/const.kpc.value
    df['Pressure (Pa)'] = pressures[:]
    df['Volume (kpc3)'] = 2*volumes[:]/const.kpc.value**3
    if model == "Falle" or model == "falle":
        df['Axis Ratio'] = np.sqrt((np.pi*lengths[:]**3)/volumes[:])
    else:
        df['Axis Ratio'] = np.sqrt((chi*lengths[:]**3)/volumes[:])

    # write data to file
    df.to_csv('LDtracks/LD_Q={:.2f}_rho={:.2f}_theta={:.2f}_model={:s}.csv'.format(jet_power, np.abs(np.log10(rho0Value)), opening_angle, model.lower()), index=False)
    

# find values of density parameter in each beta region
def __DensityParameter(nregions, k0Value, betas, regions):
        
    # instantiate variables
    kValues = np.zeros(nregions)
        
    # calculate density parameters in each region
    for count in range(0, nregions):
        # match tracks between regions `a' and `b'
        if count > 0:
            # find replicating core density in region `b' required to match pressures and times
            kValues[count] = kValues[count - 1]*regions[count]**(betas[count] - betas[count - 1])
        # if first region, set initial value of replicating core density as actual core density
        else:
            kValues[count] = k0Value
    
    return kValues


# function to apply Runge-Kutta method and extract values at requested time steps
def __runge_kutta(jet_power, source_ages, opening_angle, nregions, betas, regions, kValues, temperature, model):

    # instantiate variables
    X = np.zeros(4) # vector for time, radius, volume and pressure
    regionPointer = 0
    lengths, volumes, pressures = np.zeros_like(source_ages), np.zeros_like(source_ages), np.zeros_like(source_ages)
    
    for timePointer in range(0, len(source_ages)):
        # set initial conditions for each volume element
        if timePointer == 0:
            # calculate initial time, radius, volume and energy for the ODE, and set pointer to current region
            X[0] = limTime
            X[1] = c_speed*limTime
            X[2] = X[1]**3
            X[3] = 0
            regionPointer = 0
            # test if this radius is above start of second region boundary
            if (regions[1] < X[1]):
                X[1] = regions[1]
                X[0] = regions[1]/c_speed
                X[2] = X[1]**3
                X[3] = 0
                regionPointer = 1
            
        # solve ODE to find radius, volume and pressue at each time step
        while (X[0] < source_ages[timePointer]):
            # calculate the appropriate density profile for each angle theta
            while (regionPointer + 1 < nregions and X[1] > regions[regionPointer + 1]):
                regionPointer = regionPointer + 1

            # check if next step passes time point of interest
            if (X[0]*stepRatio > source_ages[timePointer]):
                step = source_ages[timePointer] - X[0]
            else:
                step = X[0]*(stepRatio - 1)

            # update estimates of time, radius, volume and energy
            if model == "Falle" or model == "falle":
                __falle_rk4(step, X, jet_power, opening_angle, regionPointer, betas, kValues, temperature)
            else:
                __scheuer_rk4(step, X, jet_power, opening_angle, regionPointer, betas, kValues, temperature)
             
        # record the radius, volume and pressure at current time step
        lengths[timePointer] = X[1]
        volumes[timePointer] = X[2]
        pressures[timePointer] = X[3]*(gammaC - 1)*(equipartition + 1)/X[2]
        
    return lengths, volumes, pressures
    


# Runge-Kutta method to numerical solve dynamical model over boundaries of general ambient medium
def __falle_rk4(step, X, jet_power, opening_angle, regionPointer, betas, kValues, temperature):
    
    # instantiate variables
    Y, K1, K2, K3, K4 = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    
    # fouth order Runge-Kutta method
    __falle_equations(X, K1, jet_power, opening_angle, regionPointer, betas, kValues, temperature)
    Y[:] = X[:] + 0.5*step*K1[:]
    __falle_equations(Y, K2, jet_power, opening_angle, regionPointer, betas, kValues, temperature)
    Y[:] = X[:] + 0.5*step*K2[:]
    __falle_equations(Y, K3, jet_power, opening_angle, regionPointer, betas, kValues, temperature)
    Y[:] = X[:] + step*K3[:]
    __falle_equations(Y, K4, jet_power, opening_angle, regionPointer, betas, kValues, temperature)
    X[:] = X[:] + (step/6.)*(K1[:] + 2*K2[:] + 2*K3[:] + K4[:])

def __scheuer_rk4(step, X, jet_power, opening_angle, regionPointer, betas, kValues, temperature):
    
    # instantiate variables
    Y, K1, K2, K3, K4 = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    
    # fouth order Runge-Kutta method
    __scheuer_equations(X, K1, jet_power, opening_angle, regionPointer, betas, kValues, temperature)
    Y[:] = X[:] + 0.5*step*K1[:]
    __scheuer_equations(Y, K2, jet_power, opening_angle, regionPointer, betas, kValues, temperature)
    Y[:] = X[:] + 0.5*step*K2[:]
    __scheuer_equations(Y, K3, jet_power, opening_angle, regionPointer, betas, kValues, temperature)
    Y[:] = X[:] + step*K3[:]
    __scheuer_equations(Y, K4, jet_power, opening_angle, regionPointer, betas, kValues, temperature)
    X[:] = X[:] + (step/6.)*(K1[:] + 2*K2[:] + 2*K3[:] + K4[:])
    
    
# differential equations for radius, volume and pressure
def __falle_equations(X, f, jet_power, opening_angle, regionPointer, betas, kValues, temperature):
    
    # find constants of proportionality
    omega = np.pi*(np.sin(opening_angle*np.pi/180.))**2
    epsilon = ((gammaC - 1)*(gammaX + 1)*(5 - betas[regionPointer])**3*jet_power/(18*(9*gammaC - 4 - betas[regionPointer])*kValues[regionPointer]*omega))**(1./3)
    
    # Differential equations for X[0,1,2,3] = (time, radius, volume, pressure)
    f[0] = 1.
    f[1] = 3/(5 - betas[regionPointer]) * X[1]**((betas[regionPointer] - 2)/3) * epsilon
    f[2] = 3*omega * X[1]**2 * f[1]
    f[3] = jet_power * f[0]
    
def __scheuer_equations(X, f, jet_power, opening_angle, regionPointer, betas, kValues, temperature):
    
    # find constants of proportionality
    omega = 2*np.pi*(1 - np.cos(opening_angle*np.pi/180.))
    kappa_2 = 16*np.sqrt(np.pi*np.sqrt((omega*c_speed)**3*kValues[regionPointer]))/np.sqrt((14 - 5*betas[regionPointer])*(18 - 5*betas[regionPointer])*np.sqrt(jet_power)) * np.sqrt((gammaC - 1)*(equipartition + 1)/((14 - 5*betas[regionPointer])*(gammaC - 1)*(equipartition + 1) + 2*(4 - betas[regionPointer])))
    
    # Differential equations for X[0,1,2,3] = (time, radius, volume, pressure)
    f[0] = 1.
    f[1] = X[1]**((betas[regionPointer] - 2)/2) * np.sqrt(jet_power/(omega*kValues[regionPointer]*c_speed))
    f[2] = kappa_2*((14 - 5*betas[regionPointer])/4) * X[1]**((10 - 5*betas[regionPointer])/4) * f[1]
    f[3] = (4 - betas[regionPointer])/2 * np.sqrt(jet_power*omega*kValues[regionPointer]*c_speed)/(((14 - 5*betas[regionPointer])/4*(gammaC - 1)*(equipartition + 1) + (4 - betas[regionPointer])/2)) * X[1]**((2 - betas[regionPointer])/2) * f[1]

