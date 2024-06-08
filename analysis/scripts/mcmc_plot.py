import emcee
import corner

import numpy as np
import matplotlib.pyplot as plt

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

    plt.savefig('output/mcmc/{}'.format(title))

#-------------------------------------------------------

# Get corner plots
def plot_corner(reader, discard, ndim, title):

    """
    Plot corner to see posterior distribution of parameters
    """

    flat_samples = reader.get_chain(discard=discard, thin=15, flat=True)

    # Convert radian values to degree for latitude, longitude, and inclination
    flat_samples[:,0] = np.rad2deg(flat_samples[:,0])
    flat_samples[:,1] = np.rad2deg(flat_samples[:,1])
    flat_samples[:,2] = np.rad2deg(flat_samples[:,2])

    labels = ['theta [deg]', 'phi0 [deg]', 'i [deg]', 'a', 'phi_a', 'fwhm', 't_center', 'sigma', 'amp_bump']

    # Plot corner
    corner.corner(flat_samples, labels=labels)

    plt.savefig('output/mcmc/{}'.format(title))

#-------------------------------------------------------

if __name__ == "__main__":

    ndim = 9
    discard = 10000
    
    filename = "data/chains/07_theta=10to40_25000.h5"
    reader = emcee.backends.HDFBackend(filename)

    # Plot chain
    plot_chain(reader, discard=discard, ndim=ndim, title='reader_chain.png')

    # Plot corner
    plot_corner(reader, discard=discard, ndim=ndim, title='reader_corner.png')



