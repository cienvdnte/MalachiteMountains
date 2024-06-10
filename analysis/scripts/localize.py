import emcee
import corner

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd

from astropy.constants import R_sun
from multiprocessing import Pool
from scipy.interpolate import interp1d

from funcs.model import FlareModulator
from mcmc_plot import plot_chain, plot_corner

import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = "1"

import numexpr as ne

#-----------------------------------------------------------------------------
# Non-plotting functions
# ----------------------------------------------------------------------------

# Get initials seperation
def get_sep(params):

    """
    Seperate initial variables into lists to be passed to FlareModulator. 

    Attributes:
    -----------
    theta_a : float
        flare latitude
    phi0    : float
        flare longitude
    i   : float
        inclination in radian
    a   : float
        amplitude
    phi_a   : float
        time of flare peak
    fwhm    : float
        full-width of half-maximum of flare 
    t_center: float
        center of gaussian bump
    sigma   : float
        sigma of gaussian bump
    amp_bump: float
        amplitude of gaussian bump
    
    Return:
    -----------
    flareparams: list of tuples
        flare parameters: a, phi_a, fwhm
    bumpparams: list of tuples
        bump parameters: t_cnter, sigma, amp_bump

    """

    theta_a, phi0, i, a, phi_a, fwhm, t_center, sigma, amp_bump = params

    # flare parameters
    flareparams = [(a, phi_a, fwhm)]

    # bump parameters
    bumpparams = [t_center, sigma, amp_bump]
    
    return theta_a, phi0, i, flareparams, bumpparams

# -----------------------------------------------------

# Get log priors 
def log_prior(par):

    """
    Log prior for the parameters. 

    Informed prior:
    latitude -> calculated from night length
    inclination -> from v sin i, P, and R based on Masuda and Winn
        inclination is calculated in cos i 

    Other parameters use a uniform prior.

    Attributes:
    -----------
    params  : list of tuples
        parameters for the star, flare, and bump
        theta_a, phi0, i, a, phi_a, fwhm, t_center, sigma, amp_bump
    pdf_interp: interpolation function
        interpolation to get the prior probability distribution of cos i
        the probability distribution is attained from Masuda and Winn (2020)
        (refer to cos_i.py)
    theta_mu    : float
        mu of latitude calculated from night length, assuming a gaussian
    theta_sigma : float
        std deviation of latitude calculated from night length, assuming a gaussian

    Return:
    -----------
    log prior for each time step
    
    """

    theta_a, phi0, i, a, phi_a, fwhm, t_center, sigma, amp_bump = par

    if ((0*np.pi/180) < phi0 < (360*np.pi/180) and 0 < a < 10 and 
        1.5 < phi_a < 1.7 and 0 < fwhm < (1*np.pi) and phi_a < t_center < 30 and 
        0 < sigma < 100 and 0 < amp_bump < 10
        ):

        if 0 < np.cos(i) < 1:
            if 0 < theta_a < (np.pi/2):

                # Get prior for cos i from interpolation from cos i samples
                p = pdf_interp(np.cos(i))

                # Adds up all informed (log) priors (cos i and theta)
                if p <= 0:
                    return -np.inf
                else:
                    return (np.log(p) +
                            (np.log(1.0/(np.sqrt(2*np.pi)*theta_sigma))-
                             (0.5*(theta_a-theta_mu)**2/theta_sigma**2))
                            )
            else:
                return -np.inf
        else:
            return -np.inf
        
    return -np.inf


# -----------------------------------------------------

# Get log probability
def log_probability(par, flux, flux_err):

    """
    Get log probability for sampling

    Attributes:
    -----------
    params  : list of tuples
        parameters for the star, flare, and bump
        theta_a, phi0, i, a, phi_a, fwhm, t_center, sigma, amp_bump
    
    Return:
    -----------
    log probability for each time step
    
    """

    lp = log_prior(par)

    if not np.isfinite(lp):
        return -np.inf
    
    return lp + fm.log_likelihood(par, nobump=False)

# -----------------------------------------------------

# Get walkers at different positions
def diversivy(pos, min, max, nwalkers, idx=0):

    """
    Diversivy walker initial positions to see how they converge.

    idx 0 for theta_a (latitude)

    Attributes:
    -----------
    pos : list of tuples
        theta_a, phi0, i, a, phi_a, fwhm, t_center, sigma, amp_bump
    min : int
        starting value for theta_a dispersion
    max : int
        ending value for theta_a dispersion
    nwalkers: int
        number of walkers
    idx : int
        index of which parameter that wants to be dispersed
    
    Return:
    -----------
    new pos where one of the parameters is dispersed


    """

    value = np.linspace(min, max, nwalkers)

    # To spread walker with different initial value for theta_a
    for i in range(nwalkers):
        pos[i][idx] = np.deg2rad(value[i])
    
    return pos

# -----------------------------------------------------

# Get walkers at different positions
def get_mcmc_sampling(initial, nwalkers, ndim, iter, n, dis=True,
                      min=None, max=None, continue_from_prev=False, 
                      filename=None):

    """
    Run MCMC sampling to get posterior probability distribution of all
    parameters.

    Attributes:
    -----------
    initial : list of tuples
        theta_a, phi0, i, a, phi_a, fwhm, t_center, sigma, amp_bump
    nwalkers: int
        number of walkers, preferably more than 2 x ndim
    ndim: int
        number of dimension (parameters)
    iter: int
        number of iterations
    n   : string
        for chain name filing purposes
    dis : bool
        if True, then initial position of theta_a is dispersed
        from a to b instead of moving near initial position
    continue_from_prev: bool
        if True, then chain continues from last chain
    filename: string
        file name for chains
    
    Return:
    -----------
    sampling of all parameters from MCMC

    """

    # Assign initial positions of walkers
    pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)

    # For filename export
    file_n = '{:.0f}'.format(np.rad2deg(initial[0]))

    # Disperse walkers?
    if dis:
        # To diversivy walkers position
        pos = diversivy(pos, min, max, nwalkers)
        file_n = '{}to{}'.format(min, max)

    # Set up backend filename
    if filename != None:
        filename = "data/chains/{}".format(filename)
    else:
        filename = "data/chains/{}_theta={}_{}.h5".format(n, file_n, iter)

    # Set up backend
    backend = emcee.backends.HDFBackend(filename)

    # If continue chain from previous pos becomes None because the initial
    # position is from the chain. Backend also doesn't reset
    if continue_from_prev:
        pos = None
    else:
        backend.reset(nwalkers, ndim)   # Make sure new chain starts from scratch

    # MCMC with threading
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, 
                                        backend=backend, args=(flux, flux_err))
        sampler.run_mcmc(pos, iter, progress=True, store=True)
    
    return sampler


#-----------------------------------------------------------------------------
# All plotting functions
# ----------------------------------------------------------------------------

# Get peak bump plots
def plot_bump(params, title):

    """
    Plot the bumps of the flare, from TESS lightcurve and model
    
    Attributes:
    -----------
    params  : list of tuples
        parameters for the star, flare, and bump
        theta_a, phi0, i, a, phi_a, fwhm, t_center, sigma, amp_bump
    
    Return:
    -----------
    plot of the modulated part of the flare

    """

    theta_a, phi0, i, flareparams, bumpparams = get_sep(params)
    
    modulated_flare = fm.modulated_flux(theta_a, phi0, i, flareparams, bumpparams, nobump=False)

    plt.figure(figsize=[10, 6])
    plt.plot(phi, modulated_flare, alpha=0.7, color='red', label='modulated peak-bump flare')
    plt.plot(phi, flux, '--', color='grey')
   
    plt.ylim(1,1.2)
    plt.xlabel('Time (radian)', fontsize=12)
    plt.ylabel('Normalized Flux', fontsize=12)

    plt.legend()
    plt.savefig('output/mcmc/plot/{}'.format(title))


#-------------------------------------------------------

if __name__ == "__main__":


    # Import files
    #-------------------------------------------------------
 
    # Extracted flare
    flare = np.loadtxt('data/extracted_flare.csv', delimiter=',', skiprows=1)

    phi = flare[:, 4]    # Time in radian
    flux = flare[:, 1] + 1  # Flux, normalized at 1
    flux_err = flare[:, 2]     # Error in flux

    # Cos i probability density based on Masuda and Winn
    cos_i_samples = np.loadtxt('data/mw_cosi.txt', delimiter=",")

    """"
    FlareModulator requires units in CGS. Time and angle are in radian.
    Stellar, flare, and bump parameters are converted.

    """

    # Stellar parameters
    #-------------------------------------------------------

    # Period (P_star)
    P = 0.3216891840137279 * u.d
    P = P.to(u.s) # in seconds

    # Luminosity of the star
    M_bol = 10.74
    L_sun = 4e33 * u.erg / u.s
    qlum = L_sun * np.power(10,((M_bol-4.8)/(-2.5))) # get L_star from M_bol of star
    qlum = qlum.value # in erg/s

    # Radius of the star
    R = 0.46 * R_sun
    R = R.to(u.cm)
    R_in = R.value  # in seconds

    # Inclination
    i = 64 * np.pi / 180  # in radian


    # Flare parameters (assuming the flare is a peak-bump flare)
    #-------------------------------------------------------

    # Flare amplitude
    a = 6

    # Time peak
    phi_a = 1.62    # in radian

    # Flare FWHM
    fwhm = .07 * np.pi

    # Flare latitude
    theta_a = 30 * np.pi / 180  # in radian

    # Preliminary latitude from night length (in radian)
    theta_mu = 0.63         # mu, assuming gaussian distribution             
    theta_sigma = 0.30      # sigma, assuming gaussian distribution

    # Flare longitude at t0
    phi0 = 40 * np.pi / 180     # in radian

    # Center for gaussian bump
    t_center = 7.4

    # Sigma of gaussian bump
    sigma = 6

    # Amplitude of gaussian
    amp_bump = 0.085


    # Creating probability density function for cos i
    #-------------------------------------------------------
    
    # Make histogram
    hist, bin_edges = np.histogram(cos_i_samples, bins=50, density=True)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Interpolate the PDF
    pdf_interp = interp1d(bin_centers, hist, bounds_error=False, fill_value=0)


    # Initiziation
    #-------------------------------------------------------

    # Call modulated flux
    fm = FlareModulator(phi, qlum, R_in, flux, flux_err, iscoupled=True)

    # Initial parameters
    initial = theta_a, phi0, i, a, phi_a, fwhm, t_center, sigma, amp_bump

    # Plot peak bump of initial parameters
    plot_bump(initial, 'initial_bump.png')


    # MCMC
    #-------------------------------------------------------

    # Define some samplers parameters
    #--------------------------

    nwalkers = 20               # number of walkers
    ndim = len(initial)         # number of dimension
    iter = 100                  # number of iterations
    n = '09'                    # for name filing purposes
    discard = 10000                 # how many to discard

    # If want to continue / plot from existing chains
    filename = '07_theta=10to40_25000.h5'
    
    # Get sampling from MCMC
    # sampler = get_mcmc_sampling(initial, nwalkers, ndim, iter, n, dis=False, continue_from_prev=True,
    #                             filename=filename)

    # Plot
    # plot_and_results(n, filename=filename)

    
    




















