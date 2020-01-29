#from altaipony.fakeflares import aflare

from scipy import optimize
import numpy as np

from .model import full_model, full_model_2flares, full_model_2flares2ars


def logit(function):
    '''Make a probability distribution
    a log probability distribution.'''
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        np.seterr(divide='ignore') # ignore division by zero because you want to have the -np.inf results
        result = np.log(result)
        return result
    return wrapper


@logit
def uninformative_prior(rate, minrate, maxrate):
    '''Uninformative prior for the rates.
    Uniform within [minrate, maxrate].

    Parameters:
    -------------
    rate : float

    minrate, maxrate : float
        interval in which rate is constrained

    Return:
        Prior probability
    '''
    if ((rate >= minrate) & (rate <= maxrate)):
        return 1. / (maxrate - minrate)
    else:
        return 0


@logit
def gaussian_prior(x, mu, sigma):
    '''Evaluate a normalized Gaussian function
    with mu and sigma at x. NOT TESTED.'''
    if x > np.pi/2:
        return 0
    else:
        return  1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))


def calculate_posterior_value_that_can_be_passed_to_mcmc(lp):
    '''Do some checks to make sure MCMC will work. NOT TESTED.'''
    if not np.isfinite(lp):
        return -np.inf
    if np.isnan(lp):
        return -np.inf
    else:
        return lp


def log_prior(theta, i_mu=None, i_sigma=None, phi_a_min=0,
              phi_a_max=1e9, theta_a_min=0,
              theta_a_max=np.pi/2, a_min=0, a_max=1e9,
              fwhm_min=0, fwhm_max=1e9, phi0_min=0,
              phi0_max=2*np.pi):
    """Uniform prior for start time,
    amplitude, and duration.

    - accounts for uncertainties in inclination (prior distribution=Gauss?)
    - latitude between 0 and 90 deg
    - longitude always positive (can go multiple periods into light curve)
    - FWHM always positive.
    - Amplitude always positive.

    Parameters:
    ------------
    theta : tuple
        start time, duration, amplitude
    x : array
        time array to constrain start time
    """
    phi_a, theta_a, a, fwhm, i, phi0 =  theta

    prior = (gaussian_prior(i, i_mu, i_sigma) +
             uninformative_prior(phi_a, phi_a_min, phi_a_max) +
             uninformative_prior(theta_a, theta_a_min, theta_a_max) +
             uninformative_prior(a, a_min, a_max) +
             uninformative_prior(fwhm, fwhm_min, fwhm_max) +
             uninformative_prior(phi0, phi0_min, phi0_max))

    return calculate_posterior_value_that_can_be_passed_to_mcmc(prior)




def log_likelihood(theta, phi, flux, flux_err, qlum, Fth, R, median ):
    """Log likelihood function assuming
    Gaussian uncertainties in the data points.
    SHOULDNT THIS BE POISSON? No, because flux_err is not exactly sqrt(n), 
    and above several hundred counts
    """

    phi_a, theta_a, a, fwhm, i, phi0 = theta
    model = full_model(phi_a, theta_a, a, fwhm, i, phi0=phi0,
                      phi=phi, num_pts=100, qlum=qlum,
                      Fth=Fth, R=R, median=median)

    fr2 = flux_err**2
    val = -0.5 * np.sum((flux - model) ** 2 / fr2 + np.log(fr2))
    return val


def log_probability(theta, phi, flux, flux_err, qlum, Fth, R, median, kwargs):
    """Posterior probability to pass to MCMC sampler.
    """
    lp = log_prior(theta, **kwargs)

    if not np.isfinite(lp):
        return -np.inf
    
    try:
        ll = log_likelihood(theta, phi, flux, flux_err, qlum, Fth, R, median)
        
    except:
        return -np.inf
    
    if np.isnan(ll):
        return -np.inf
    
    return lp + ll

#-------------------------------------------------------------------------------------
# You can choose an uninformative prior if you don't know the inclination and still try
# to fit the data.

def log_prior_no_incl(theta, i_mu=None, i_sigma=None, phi_a_min=0,
                      phi_a_max=1e9, theta_a_min=0,
                      theta_a_max=np.pi/2, a_min=0, a_max=1e9,
                      fwhm_min=0, fwhm_max=1e9, phi0_min=0,
                      phi0_max=2*np.pi):
    """Uniform prior for start time,
    amplitude, and duration.

    - no information on inclination
    - latitude between 0 and 90 deg
    - longitude always positive (can go multiple periods into light curve)
    - FWHM always positive.
    - Amplitude always positive.

    Parameters:
    ------------
    theta : tuple
        start time, duration, amplitude
    x : array
        time array to constrain start time
    """
    phi_a, theta_a, a, fwhm, i, phi0 =  theta

    prior = (uninformative_prior(i, 0, np.pi) +
             uninformative_prior(phi_a, phi_a_min, phi_a_max) +
             uninformative_prior(theta_a, theta_a_min, theta_a_max) +
             uninformative_prior(a, a_min, a_max) +
             uninformative_prior(fwhm, fwhm_min, fwhm_max) +
             uninformative_prior(phi0, phi0_min, phi0_max))

    return calculate_posterior_value_that_can_be_passed_to_mcmc(prior)

def log_probability_no_incl(theta, phi, flux, flux_err, qlum, Fth, R, median, kwargs):
    """Posterior probability to pass to MCMC sampler.
    """
    lp = log_prior_no_incl(theta, **kwargs)

    if not np.isfinite(lp):
        return -np.inf
   
    try:
        ll = log_likelihood(theta, phi, flux, flux_err, qlum, Fth, R, median)
    
    except:
        return -np.inf
    
    if np.isnan(ll):
        return -np.inf
    
    return lp + ll

#------------------ TWO-FLARE MODEL ------------------------------------------------------------

def log_prior_2flares(theta, i_mu=None, i_sigma=None, phi_a_min=(0,0),
              phi_a_max=(1e9,1e9), theta_a_min=0,
              theta_a_max=np.pi/2, a_min=(0,0), a_max=(1e9,1e9),
              fwhm_min=(0,0), fwhm_max=(1e9,1e9), phi0_min=0,
              phi0_max=2*np.pi):
    """Uniform prior for start time,
    amplitude, and duration.

    - accounts for uncertainties in inclination (prior distribution=Gauss?)
    - latitude between 0 and 90 deg
    - longitude always positive (can go multiple periods into light curve)
    - FWHM always positive.
    - Amplitude always positive.

    Parameters:
    ------------
    theta : tuple
        start time, duration, amplitude
    x : array
        time array to constrain start time
    """
    phi_a1, phi_a2, theta_a, a1, a2, fwhm1, fwhm2, i, phi0 =  theta

    prior = (gaussian_prior(i, i_mu, i_sigma) +
             uninformative_prior(phi_a1, phi_a_min[0], phi_a_max[0]) +
             uninformative_prior(phi_a2, phi_a_min[1], phi_a_max[1]) +
             uninformative_prior(theta_a, theta_a_min, theta_a_max) +
             uninformative_prior(a1, a_min[0], a_max[0]) +
             uninformative_prior(fwhm1, fwhm_min[0], fwhm_max[0]) +
             uninformative_prior(a2, a_min[1], a_max[1]) +
             uninformative_prior(fwhm2, fwhm_min[1], fwhm_max[1]) +
             uninformative_prior(phi0, phi0_min, phi0_max))

    return calculate_posterior_value_that_can_be_passed_to_mcmc(prior)



def log_probability_2flares(theta, phi, flux, flux_err, qlum, Fth, R, median, kwargs):
    """Posterior probability to pass to MCMC sampler.
    """
    lp = log_prior_2flares(theta, **kwargs)

    if not np.isfinite(lp):
        print("INF PRIOR")
        return -np.inf
    
    try:
        ll = log_likelihood_2flares(theta, phi, flux, flux_err, qlum, Fth, R, median)
        
    except:
        print("LOG L ERROR")
        return -np.inf
    
    if np.isnan(ll):
        print("LOG L NAN")
        return -np.inf
    
    return lp + ll

def log_likelihood_2flares(theta, phi, flux, flux_err, qlum, Fth, R, median ):
    """Log likelihood function assuming
    Gaussian uncertainties in the data points.
    SHOULDNT THIS BE POISSON?
    """

    phi_a1, phi_a2, theta_a, a1, a2, fwhm1, fwhm2, i, phi0 =  theta
    model = full_model_2flares((phi_a1,phi_a2), theta_a, (a1,a2), (fwhm1,fwhm2), 
                               i, phi0=phi0,
                                phi=phi, num_pts=100, qlum=qlum,
                                Fth=Fth, R=R, median=median)
   # print(model)
    
    fr2 = flux_err**2
    val = -0.5 * np.sum((flux - model) ** 2 / fr2 + np.log(fr2))
    return val

#------------------ TWO-FLARE TWO ACTIVE REGIONS MODEL ------------------------------------------------------------

def log_prior_2flares2ars(theta, i_mu=None, i_sigma=None, phi_a_min=(0,0),
              phi_a_max=(1e9,1e9), theta_a_min=0,
              theta_a_max=np.pi/2, a_min=(0,0), a_max=(1e9,1e9),
              fwhm_min=(0,0), fwhm_max=(1e9,1e9), phi0_min=0,
              phi0_max=2*np.pi):
    """Uniform prior for start time,
    amplitude, and duration.

    - accounts for uncertainties in inclination (prior distribution=Gauss?)
    - latitude between 0 and 90 deg
    - longitude always positive (can go multiple periods into light curve)
    - FWHM always positive.
    - Amplitude always positive.

    Parameters:
    ------------
    theta : tuple
        start time, duration, amplitude
    x : array
        time array to constrain start time
    """
    phi_a1, phi_a2, theta_a1, theta_a2, a1, a2, fwhm1, fwhm2, i, phi0 =  theta

    prior = (gaussian_prior(i, i_mu, i_sigma) +
             uninformative_prior(phi_a1, phi_a_min[0], phi_a_max[0]) +
             uninformative_prior(phi_a2, phi_a_min[1], phi_a_max[1]) +
             uninformative_prior(theta_a1, theta_a_min, theta_a_max) +
             uninformative_prior(theta_a2, theta_a_min, theta_a_max) +
             uninformative_prior(a1, a_min[0], a_max[0]) +
             uninformative_prior(fwhm1, fwhm_min[0], fwhm_max[0]) +
             uninformative_prior(a2, a_min[1], a_max[1]) +
             uninformative_prior(fwhm2, fwhm_min[1], fwhm_max[1]) +
             uninformative_prior(phi0, phi0_min, phi0_max))

    return calculate_posterior_value_that_can_be_passed_to_mcmc(prior)



def log_probability_2flares2ars(theta, phi, flux, flux_err, qlum, Fth, R, median, kwargs):
    """Posterior probability to pass to MCMC sampler.
    """
    lp = log_prior_2flares2ars(theta, **kwargs)

    if not np.isfinite(lp):
        print("INF PRIOR")
        return -np.inf
    
    try:
        ll = log_likelihood_2flares2ars(theta, phi, flux, flux_err, qlum, Fth, R, median)
        
    except Exception as e:
        print("LOG L ERROR")
        print(e)
        return -np.inf
    
    if np.isnan(ll):
        print("LOG L NAN")
        return -np.inf
    
    return lp + ll

def log_likelihood_2flares2ars(theta, phi, flux, flux_err, qlum, Fth, R, median ):
    """Log likelihood function assuming
    Gaussian uncertainties in the data points.
    SHOULDNT THIS BE POISSON?
    """

    phi_a1, phi_a2, theta_a1, theta_a2, a1, a2, fwhm1, fwhm2, i, phi0 =  theta
    
    model = full_model_2flares2ars((phi_a1,phi_a2), (theta_a1, theta_a2), (a1,a2), (fwhm1,fwhm2), 
                               i, phi0=phi0,
                                phi=phi, num_pts=100, qlum=qlum,
                                Fth=Fth, R=R, median=median)
   # print(model)
    
    fr2 = flux_err**2
    val = -0.5 * np.sum((flux - model) ** 2 / fr2 + np.log(fr2))
    return val
