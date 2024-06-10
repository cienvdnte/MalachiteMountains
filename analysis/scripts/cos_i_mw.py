import emcee
import corner

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from astropy.constants import R_sun
from scipy.interpolate import interp1d

from uncertainties import ufloat
from uncertainties.umath import *

#-------------------------------------------------------

# Get Gaussian function
def gauss_func(x, mu, sigma):

    """
    Gaussian function

    Attributes:
    -----------
    x   : float
        value to be evaluated
    mu  : float
        mean of gaussian distribution (value from observation)
    sigma: float
        standard deviation of gaussian distribution (error from observation)

    Return:
    -----------
    probability from gaussian distribution: float

    """

    nom = np.exp(-1.0 * np.power(x - mu,2) / (2 * np.power(sigma,2)))
    den = np.sqrt(2*np.pi) * sigma

    return nom/den

#-------------------------------------------------------

# Get likelihood function 
def log_likelihood(par):

    """
    Getting log likelihood for MCMC.
    Log likelihood is the gaussian probability of the parameter in reference
    to values attained from observations.

    This is based on Masuda and Winn (2020)

    Attributes:
    -----------
    par  : list of tuples
        parameters of the star
    R_star: float
        radius of the star
    P_star: float
        period of the star
    cos_i : float
        cos of stellar inclination

    Return:
    -----------
    sum of log likelihood of all parameters

    """

    # Initiate sum of log probability
    log_prob = 0.0

    # Read parameter
    R_star, P_star, cos_i = par

    # Determine log likelihood of R
    R_likelihood = gauss_func(R_star, R_obs, sigma_R_obs)

    # Determine log likelihood of P
    P_likelihood = gauss_func(P_star, P_obs, sigma_P_obs)

    # Find v_eq from v = 2piR/P
    nom = 2 * np.pi * R_star
    den = P_star
    v_eq = nom/den

    # Making sure 1 - cos^2 i is more than 0
    term_inside_sqrt = np.abs(1-cos_i**2)

    # Calculate v sin i from R and P
    vsini = v_eq*np.sqrt(term_inside_sqrt)

    # Determine log likelihood of v sin i
    vsini_likelihood = gauss_func(vsini, vsini_obs, sigma_vsini_obs)

    # Sum of all probability
    log_prob = np.log(R_likelihood) + np.log(P_likelihood) + np.log(vsini_likelihood)

    if log_prob != 0:
        return log_prob
    else:
        return -np.inf

#-------------------------------------------------------

# Get prior function
def log_prior(par):

    """
    Prior distribution for parameters.

    Because prior of R, P, and v sin i are gaussian,
    they are calculated in likelihood.

    Cos i should be between 0 and 1.

    Attributes:
    -----------
    par  : list of tuples
        parameters of the star
    R_star: float
        radius of the star
    P_star: float
        period of the star
    cos_i : float
        cos of stellar inclination

    Return:
    -----------
    0 if parameter is within boundary
    -inf if not

    """

    R_star, P_star, cos_i = par

    if 0 < cos_i < 1:
        return 0.0
    else:
        return -np.inf

#-------------------------------------------------------

# Get prior function
def log_probability(par):

    """
    Log probability of all of the parameters at every step.

    Attributes:
    -----------
    par  : list of tuples
        parameters of the star
    R_star: float
        radius of the star
    P_star: float
        period of the star
    cos_i : float
        cos of stellar inclination

    Return:
    -----------
    Sum of log prior and log likelihood

    """

    # Get log prior
    lp = log_prior(par)

    if not np.isfinite(lp):
        return -np.inf
    
    return log_prior(par) + log_likelihood(par)

#-------------------------------------------------------

# Get chains
def plot_chain(sampler, discard, ndim, labels, title):

    """
    Plot chains for sampler

    Attributes:
    -----------
    sampler : array-like
        samplers from mcmc
    discard : integer
        numbers of chain to cut off
    ndim    : integer
        number of dimension
    labels  : array-like
        name of labels
    title   : string
        title to save in png

    """

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

    # Get all samples to be plotted with several iterations discarded
    samples = sampler.get_chain(discard=discard)

    # Plot the chains for each parameter
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    plt.savefig('output/mcmc/{}'.format(title))

#-------------------------------------------------------

# Get corner plot
def plot_corner(sampler, discard, ndim, labels, title):

    """
    Plot posterior distribution of each parameteres using corner plot

    Attributes:
    -----------
    sampler : array-like
        samplers from mcmc
    discard : integer
        numbers of chain to cut off
    ndim    : integer
        number of dimension
    labels  : array-like
        name of labels
    title   : string
        title to save in png

    """

    # Get all samples to be plotted with several iterations discarded
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)

    # Plot a corner plot
    fig = corner.corner(flat_samples, labels=labels)

    plt.savefig('output/mcmc/{}'.format(title))

#-------------------------------------------------------

# Get analytical solution for posterior distribution
def posterior_cosi(params, cos_i):

    """
    Getting the analytical posterior distribution for cos i.

    Formula from Bowler et. al. (2023) (https://ui.adsabs.harvard.edu/abs/2023AJ....165..164B/abstract)
    Equation A11 adapted from Masuda and Winn (2020)

    Attributes:
    -----------
    params : array-like
        parameters of the star
        vsini, sigma_visini, v_eq, sigma_veq

    Return:
    -----------
    analytical posterior distribution of cos i
    """

    # define parameters from params
    vsini, sigma_vsini, v_eq, sigma_veq, = params

    # Calculate the analytical posterior distribution
    upper_e = vsini - ((v_eq) * np.sqrt(1 - (cos_i**2)))
    below_e = 2 * ((sigma_vsini**2) + ((sigma_veq**2) * (1 - (cos_i**2))))

    below = np.sqrt((sigma_vsini**2) + ((sigma_veq**2) * (1 - (cos_i**2))))

    return np.exp(-(np.power(upper_e,2)/below_e)) / below

#-------------------------------------------------------

# Plot analytical solution
def plot_analytical(param, cos_i_samples, scale, title):

    """
    Getting the analytical posterior distribution for cos i.

    Formula from Masuda and Winn (2020).

    Attributes:
    -----------
    params : list of tuples
        parameters of the star
        vsini, sigma_visini, v_eq, sigma_veq
    cos_i_samples: array-like
        samples of cos i from MCMC
    scale   : int
        an arbitary number to scale analytical posterior distribution
        in the same order of the samples from MCMC.
        both are not normalized so arbitary scaling doesn't have any
        physical meaning.
    title   : string
        name file

    Return:
    -----------
    plot of analytical posterior distribution of cos i
    """
    
    # Make histogram
    hist, bin_edges = np.histogram(cos_i_samples, bins=50, density=True)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Number of random angles to generate
    num_samples = 10000

    # Generate random angles between 0 and pi (180 degrees)
    cos_i = np.linspace(0, 1, num_samples)

    # Get analytical posterior
    pos_cos = posterior_cosi((param[0], param[1], param[6], param[7]), cos_i)

    # Interpolate the PDF
    pdf_interp = interp1d(bin_centers, hist, bounds_error=False, fill_value=0)

    # Plot a histogram of the marginal likelihood of cos_i
    plt.figure(figsize=[8,6])

    plt.hist(cos_i_samples, bins=50, density=True, color='green', 
             edgecolor='black', alpha=0.5, label='MCMC Marginal Posterior')
    
    plt.plot(cos_i, pos_cos*scale, color='blue', linewidth=5, alpha=0.5, 
             label='Analytical Marginal Posterior')
    
    plt.plot(cos_i, pdf_interp(cos_i), color='red', linewidth=5, alpha=0.5,
              label='Interpolation Marginal Posterior')

    # Add labels and title
    plt.xlabel('cos_i')
    plt.ylabel('Density')
    plt.title('v sin i = {} +/- {} km/s, v_eq = {} +/- {} km/s'.format(param[0], param[1], param[6], param[7]))
    plt.xlim(0,1)
    # plt.ylim(0,8)

    plt.legend()

    plt.savefig('output/mcmc/{}'.format(title))

#-------------------------------------------------------

# Plot analytical solution
def plot_i_samples(cos_i_samples, title):

    """
    Getting the analytical posterior distribution for cos i.

    Formula from Masuda and Winn (2020).

    Attributes:
    -----------
    cos_i_samples: array-like
        samples of cos i from MCMC
    title   : string
        name file

    Return:
    -----------
    plot of analytical posterior distribution of cos i
    """

    # Comvert cos i to i in deg
    i_samples = np.rad2deg(np.arccos(cos_i_samples))

    # Find 16th, 50th, 84th percentile (68% C.I)
    mcmc = np.percentile(i_samples, [16, 50, 84])
    q = np.diff(mcmc)

    # Plot a histogram of the marginal likelihood of i
    plt.figure(figsize=[8,6])
    N, _, patches = plt.hist(i_samples, bins=50, density=True)
    
    # Label patches based on 68% C.I
    for i in range(len(N)):

        patches[i].set_facecolor('royalblue')
        patches[i].set_edgecolor('black')

        xleft = patches[i].get_x()
        xright = xleft + patches[i].get_width()

        if (xleft >= mcmc[0]) and (xright <= mcmc[2]):
            patches[i].set_alpha(0.7)
        else:
            patches[i].set_alpha(0.4)

    # Create custom legend labels
    highlight_patch = mpatches.Patch(alpha=0.7, color='royalblue', label='68% C.I')
    normal_patch = mpatches.Patch(alpha=0.4, color='royalblue', label='Marginal posterior')

    # Add labels and title
    plt.xlabel('i [deg]')
    plt.ylabel('Density')
    plt.title('i = {:.2f} (+{:.2f}/-{:.2f}) [deg]'.format(mcmc[1], q[1], q[0]))
    plt.xlim(0,90)
    # plt.ylim(0,8)

    # Add legend to the plot
    plt.legend(handles=[highlight_patch, normal_patch])

    # plt.legend()

    plt.savefig('output/mcmc/{}'.format(title))

#-------------------------------------------------------

if __name__ == "__main__":

    # Parameters
    #-------------------------------------------------------

    # Parameters -> vsini, sigma_vsini_obs, P_obs, sigma_P_obs, 
    #               R_obs, sigma_R_obs, v_eq_obs, sigma_veq_obs
    
    # v_eq_obs is not calculated directly from P_obs and R_obs so that
    # values from Masuda and Winn can be used in the program for benchmarking

    param = [68.12, 3.77, 0.3216891840137279, 0.3216891840137279*0.05, 
             0.48, 0.04, 75.57, 6.30] # my star

    # v sin i
    vsini_obs = param[0]
    sigma_vsini_obs = param[1]             # in km/s

    # Period of the star
    P_obs = param[2] * u.d
    P_obs = P_obs.to(u.s)
    P_obs = P_obs.value                    # in s

    # Standard deviation of P_star
    sigma_P_obs = param[3] * u.d
    sigma_P_obs = sigma_P_obs.to(u.s)
    sigma_P_obs = sigma_P_obs.value        # in s

    # Radius of the star
    R_obs = param[4] * R_sun 
    R_obs = R_obs.to(u.km)
    R_obs = R_obs.value                    # in km

    # Standard deviation of R_star
    sigma_R_obs = param[5] * R_sun 
    sigma_R_obs = sigma_R_obs.to(u.km)
    sigma_R_obs = sigma_R_obs.value         # in km

    # Equatorial velocity
    v_eq_obs = param[6]                     # in km/s
    sigma_veq_obs = param[7]                # in km/s

    # Initial values
    cos_i = np.cos(np.deg2rad(50))

    initial = np.array([[R_obs, P_obs, cos_i]])  # Make sure it's a 2D array

    # MCMC
    #-------------------------------------------------------

    # Initialize walkers and dimensions
    pos = initial + 1e-3 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    # Run MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(pos, 50000, progress=True)

    # Name of labels
    labels = ["R_star", "P_star", "cos_i"]

    # How many iterations to be discarded
    discard = 0

    # Plot chain
    plot_chain(sampler, discard, ndim, labels, title='cosi_chain.png')

    # Plot corner
    plot_corner(sampler, discard, ndim, labels, title='cosi_corner.png')

    # Get the chain from the sampler
    chain = sampler.get_chain(flat=True)

    # Extract the relevant parameter values (in this case, the first parameter is cos_i)
    cos_i_samples = chain[:, 2]

    # Export cos_i
    np.savetxt('data/mw_cosi_2.txt', cos_i_samples, delimiter=',')

    # Plot analytical
    plot_analytical(param, cos_i_samples, scale=13.5, title='mw_cosi.png')

    # Plot i_samples
    plot_i_samples(cos_i_samples, title='mw_inclination.png')













