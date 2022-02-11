import numpy as np
from model import all_peaks_model
import time

# functions used by the MCMC sampler
def log_probability(theta, *args):
    """the probability function. This is directly called by emcee, the parameters that are varied are in theta."""
    freq, power, fit_info = args

    fit_info.set_all_fit_variables(theta)
    lp = log_prior(theta, fit_info)
    # if log of prior is neg infinity then full log probability function must be too
    if not np.isfinite(lp):
        return -np.inf
    return_value = lp + log_likelihood(freq, power, fit_info)
    return return_value


def log_prior(thetas, fit_info):
    """ prior function called by log_probability().
    Returns negative infinity if any of the parameter values
        are outside the priors. Uses uniform priors currently."""
    for (theta, lower, upper) in zip(thetas, fit_info.lower_prior, fit_info.upper_prior):
        if not (lower < theta < upper):
            return -np.inf
    return 0

def log_likelihood(freq, measured, fit_info):
    model = all_peaks_model(freq, fit_info)
    return_value = -np.sum(np.log(model) + measured / model)
    return return_value
