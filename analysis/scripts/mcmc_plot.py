import emcee
import corner

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.constants import R_sun
from funcs.model import FlareModulator

#-------------------------------------------------------

# Get chain plots
def plot_chain(reader, discard, ndim, title):

    """
    Plot chains of walkers for n iteration
    """

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

    # Get chain from reader/sampler
    samples = reader.get_chain(discard=discard)
    
    labels = ["theta", "phi0", "i", "a", "phi_a", "fwhm", "t_center", "sigma", "amp_bump"]

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    plt.savefig('output/mcmc/plot{}'.format(title))

#-------------------------------------------------------

# Get corner plots
def plot_corner(reader, discard, title):

    """
    Plot corner to see posterior distribution of parameters
    """

    flat_samples = reader.get_chain(discard=discard, thin=15, flat=True)

    # Convert radian values to degree for latitude, longitude, and inclination
    flat_samples[:,0] = np.rad2deg(flat_samples[:,0])
    flat_samples[:,1] = np.rad2deg(flat_samples[:,1])
    flat_samples[:,2] = np.rad2deg(flat_samples[:,2])

    labels = ['theta [deg]', 'phi0 [deg]', 'i [deg]', 'a', 'phi_a [rad]', 'fwhm', 't_center [rad]', 'sigma', 'amp_bump']

    # Plot corner
    corner.corner(flat_samples, labels=labels)

    plt.savefig('output/mcmc/plot{}'.format(title))

#-------------------------------------------------------

# Get plot of flare fits
def plot_flarefit(reader, discard, title):

    """
    Plot flare fit 

    """

    plt.figure(figsize=[14,8])

    flat_samples = reader.get_chain(discard=discard, thin=15, flat=True)

    inds = np.random.randint(len(flat_samples), size=200)

    for ind in inds:
        sample = flat_samples[ind]

        flareparams = [(sample[3], sample[4], sample[5])]
        bumpparams = [sample[6], sample[7], sample[8]]

        plt.plot(phi, fm.modulated_flux(sample[0], sample[1], sample[2], flareparams, 
                                        bumpparams, nobump=False), 
                color='red', alpha=0.5)
        
        plt.plot(phi, fm.bump_template(flareparams[0], bumpparams)+1, color='grey', alpha=0.1)

    plt.scatter(phi, flux, label='TIC 206544316', color='black', s=10)
    plt.ylim(1,1.2)

    plt.xlabel('Time (radian)', fontsize=12)
    plt.ylabel('Normalized flux', fontsize=12)

    plt.legend()

    plt.savefig('output/mcmc/plot{}'.format(title))

#-------------------------------------------------------

# Get plots and results
def plot_and_results(sampler=None, filename=None):

    """
    Plot corner to see posterior distribution of parameters
    """

    # Plot fit and get results
    if filename != None:
        sampler = emcee.backends.HDFBackend('data/chains/{}'.format(filename))
    else:
        sampler = sampler

    # #Plot chains
    # plot_chain(sampler, discard, ndim, title='sampler_chain_{}.png'.format(n))

    # # Plot corner plot
    # plot_corner(sampler, discard, title='sampler_corner_{}.png'.format(n))

    # Plot flarefit
    # plot_flarefit(sampler, discard, title='sampler_flarefit_{}.png'.format(n))

#-------------------------------------------------------

if __name__ == "__main__":

    # Import files
    #-------------------------------------------------------
 
    # Extracted flare
    flare = np.loadtxt('data/extracted_flare.csv', delimiter=',', skiprows=1)

    phi = flare[:, 4]    # Time in radian
    flux = flare[:, 1] + 1  # Flux, normalized at 1
    flux_err = flare[:, 2]     # Error in flux

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

    # Call modulated flux
    fm = FlareModulator(phi, qlum, R_in, flux, flux_err, iscoupled=True)

    # MCMC parameters
    #-------------------------------------------------------

    ndim = 9
    discard = 0

    filename = '07_theta=10to40_25000.h5'
    reader = emcee.backends.HDFBackend(filename)

    # Plot chain
    plot_chain(reader, discard, ndim, title='reader_chain.png')

    # Plot corner
    plot_corner(reader, discard, title='reader_corner.png')

    #-------------------------------------------------------







