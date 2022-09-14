""" probability_functions.py
This contains the function log_probability() which is sampled by the EMCEE sampler.
The log(probability) function is created as a combination of a prior probability and a likelihood from the
log_likelihood function."""
import numpy as np
from model import all_peaks_model
import time

def log_probability(theta, *args):
    """
    The probability function that is directly called by emcee, the parameters that are varied are in theta.
    :param theta: The list of parameters that are varied.
    :param args: The other required information to find the probability: the frequency and power arrays,
                and fit_info. fit_info in an instance of class Fit_Info and
                contains all the parameter values for the parameters that aren't being varied.
                It also contains the bounds of the parameters which is used to calculate
                the prior probability for each parameter.
    :return: The log_probability for the particular parameters in theta.
    """
    freq, power, fit_info = args

    # put all the new fit parameter values into the fit_info structure
    fit_info.set_all_fit_variables(theta)
    # calculate the log prior probability.
    lp = log_prior(theta, fit_info)
    # if log of prior is neg infinity then full log probability function must be too
    if not np.isfinite(lp):
        return -np.inf
    # total log probability is addition of log prior and log likelihood
    return_value = lp + log_likelihood(freq, power, fit_info)
    return return_value


def log_prior(thetas, fit_info):
    """
    prior function called by log_probability().
    Returns negative infinity if any of the parameter values are outside the priors, else returns 0.
    Uses uniform priors.
    :param thetas: list of fit parameters
    :param fit_info: instance of class Fit_Info which contains the upper and lower bounds on each parameter
    :return:
    """

    """ """
    for (theta, lower, upper) in zip(thetas, fit_info.lower_prior, fit_info.upper_prior):
        if not (lower < theta < upper):
            return -np.inf
    return 0

def log_likelihood(freq, measured, fit_info):
    """
    returns the log_likelihood for the set of fit parameters.
    :param freq: the frequency array
    :param measured: the measured data (i.e. from GOLF). This is the power array.
    :param fit_info: instance of class Fit_Info which contains the values of all the parameters
    required to generate the model.
    :return: the log_likelihood for the fit parameters.
    """
    # generate the model made up of Lorentzians for the new fit parameters
    model = all_peaks_model(freq, fit_info)
    # chi squared to with two degrees of freedom log likelihood
    return_value = -np.sum(np.log(model) + measured / model)
    return return_value
