import emcee
import corner

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.constants import R_sun
from funcs.model import FlareModulator

#-------------------------------------------------------

def convert_to_deg(sam):

    """
    Convert inclination, longitude, and latitude from rad to deg
    """

    # Convert radian values to degree for latitude, longitude, and inclination
    sam[:,0] = np.rad2deg(sam[:,0])
    sam[:,1] = np.rad2deg(sam[:,1])
    sam[:,2] = np.rad2deg(sam[:,2])

    return sam

#-------------------------------------------------------

# Get chain plots
def plot_chain(sampler, title):

    """
    Plot chains of walkers for n iteration

    """
    # Get samples from MCMC
    samples = sampler.get_chain(discard=discard)

    # Get ndim
    ndim = samples.shape[2]

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    
    labels = ["theta", "phi0", "i", "a", "phi_a", "fwhm", "t_center", "sigma", "amp_bump"]

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    plt.savefig('output/mcmc/plot/{}'.format(title))

#-------------------------------------------------------

# Get corner plots
def plot_corner(sampler, discard, labels, title):

    """
    Plot corner to see posterior distribution of parameters
    """

    # Flat samples
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)

   
    # Plot corner
    corner.corner(flat_samples, labels=labels)

    plt.savefig('output/mcmc/plot/{}'.format(title))

#-------------------------------------------------------

# Get plot of flare fits
def plot_flarefit(sampler, discard, title):

    """
    Plot flare fit 

    """
    # Flat samples
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)

    # Get random index to sample from flat_samples
    inds = np.random.randint(len(flat_samples), size=200)

    fig, ax = plt.subplots(2, 1, figsize=[15,8], gridspec_kw={'height_ratios': [3, 1]})

    for idx, ind in enumerate(inds):

        # Get random sample with index ind
        sample = flat_samples[ind]

        # Get parameter of flare and peak bump
        flareparams = [(sample[3], sample[4], sample[5])]
        bumpparams = [sample[6], sample[7], sample[8]]

        # Modulated flare from sampled parameters
        flare_mod = fm.modulated_flux(sample[0], sample[1], sample[2], flareparams, 
                                            bumpparams, nobump=False)
        
        # Underlying flare 
        flare_bump = fm.bump_template(flareparams[0], bumpparams)+1

        # Residual
        residual = flux - flare_mod

        # To print with proper labels
        if idx == (len(inds) - 1):
            ax[0].plot(time, flare_mod, color='red', alpha=0.5, label='modulated flux')
            ax[0].plot(time, flare_bump, color='grey', alpha=0.1, label='underlying flare')

            ax[1].plot(time, residual, color='grey', label='residual')
            
        else:
            ax[0].plot(time, flare_mod, color='red', alpha=0.5)
            ax[0].plot(time, flare_bump, color='grey', alpha=0.1)

            ax[1].plot(time, residual)

    ax[0].scatter(time, flux, label='TIC 206544316', color='black', s=10)

    ax[0].set_ylim(1,1.2)
    ax[0].set_xlim(min(time), max(time))
    ax[1].set_xlim(min(time), max(time))

    ax[0].set_ylabel('Normalized flux', fontsize=12)
    ax[1].set_xlabel('Time - 2457000 (BTJD)', fontsize=12)
    ax[1].set_ylabel('Residual', fontsize=12)

    ax[0].legend()
    ax[1].legend()

    plt.savefig('output/mcmc/plot/{}'.format(title))

    #------------------------------------
    
    # To also plot the peak
    ax[0].set_yscale('log')
    ax[0].set_yticks([1, 2, 3, 4, 6])
    ax[0].get_yaxis().set_major_formatter(plt.ScalarFormatter())

    ax[0].set_ylabel('Log normalized flux', fontsize=12)
    ax[0].set_ylim()

    plt.savefig('output/mcmc/plot/log_{}'.format(title))


#-------------------------------------------------------

# Get plot of flare fits
def print_results(sampler, discard, labels):

    # Get flat samples
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)

    # Get ndim
    ndim = flat_samples.shape[1]

    # Convert to deg
    flat_samples = convert_to_deg(flat_samples)

    # Export results in a .txt file
    # Clear the file before starting the loop
    with open('output/mcmc/mcmc_sampling.txt', 'w') as f:
        f.write('-------- parameters from MCMC sampling --------\n')  

    for i in range(ndim):

        # Find 16th, 50th, 84th percentile. This is 68% C.I
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])

        # Find different between 16th-50th, and 50th-84th
        q = np.diff(mcmc)

        # Open file in append mode
        with open('output/mcmc/mcmc_sampling.txt', 'a') as f:
            f.write('{} = {:.4f} (+{:.4f}, -{:.4f})\n'.format(labels[i], mcmc[1], q[0], q[1]))

#-------------------------------------------------------

# Get plots and results
def plot_and_results(sampler=None, filename=None):

    """
    Plot chain, corner, and flarefit
    """

    # Plot fit and get results
    if filename != None:
        sampler = emcee.backends.HDFBackend('data/chains/{}'.format(filename))
    else:
        sampler = sampler
    
    # Samples

    #Plot chains
    plot_chain(sampler, discard, title='reader_chain.png')

    # Define labels for corner, flare fit, and print
    labels = ['theta [deg]', 'phi0 [deg]', 'i [deg]', 'a', 'phi_a [rad]', 'fwhm', 't_center [rad]', 'sigma', 'amp_bump']

    # Plot corner plot
    plot_corner(sampler, discard, labels, title='reader_corner.png')

    # Plot flarefit
    plot_flarefit(sampler, discard, title='reader_flarefit.png')

    # Export results
    print_results(sampler, discard, labels)



#-------------------------------------------------------

if __name__ == "__main__":

    # Import files
    #-------------------------------------------------------
 
    # Extracted flare
    flare = np.loadtxt('data/longer_extracted_flare.csv', delimiter=',', skiprows=1)

    time = flare[:, 0]      # Time in BJD
    phi = flare[:, 4]       # Time in radian
    flux = flare[:, 1] + 1  # Flux, normalized at 1
    flux_err = flare[:, 2]  # Error in flux


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

    discard = 10000
    filename = '07_theta=10to40_25000.h5'

    plot_and_results(filename=filename)

    #-------------------------------------------------------







