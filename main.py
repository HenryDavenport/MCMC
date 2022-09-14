""" main.py
Runs Markov Chain Monte Carlo Sampling using emcee library on Helioseismology "Sun as a star" data.
"""
# import standard libraries
import math
import multiprocessing
import os
import time
import copy
# import project specific external libraries
import corner
import emcee
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from dataclasses import dataclass
# import internal python files
from import_data import import_data
import model
from probability_functions import log_probability
from structures import Fit_Info, Parameter
from run_settings import Run_Settings
from import_priors import import_priors, find_pairs
from initialise_walker_positions import walker_pos
from output import create_directory, save_output

# Set the range of values explored for the background of the fits.
# The lower and upper bounds set the range of values explored by the MCMC walkers.
lower_bound_background = 0
expected_background_value = 50000  # the approximate expected value of the background in the fit
upper_bound_background = 100000
background = Parameter(expected_background_value, lower_bound_background, upper_bound_background,
                       constant=False, shared=False)

# run_settings is a structure which contains settings for the fits
# i.e. not just information about one particular pair of peaks
# use to edit window width, mode visibilities, range of peaks fitted etc.
run_settings = Run_Settings(background)

# create directory to save results in. Input to function sets beginning of folder name
folder_identifier = "GOLF"
directory = create_directory(folder_identifier)

# file (in same directory as main.py) with time series data that will be fitted
filename = "GOLF_bw_rw_pm3_960411_200830_Sync_RW_dt60F_v1.fits"

# the file in which all the data about the peaks is stored -
# the predicted values of all parameters and the range in which
# each parameter value will be.
priors_filename = "priors.txt"

# Example: import 1 years data and FFT the data
no_datapoints_for_year = 31536000 / run_settings.cadence
start_year = 0  # start year from beginning of data for fit
end_year = 1  # end year from beginning of data for fit

# set the range of datapoints in the time series based on the start and end year
start_year_index = start_year*no_datapoints_for_year # starting datapoint index
end_year_index = end_year*no_datapoints_for_year # end datapoint index

# import data and return fourier transformed time series (power) and frequency (freq) array
power, freq = import_data(filename, start_year_index, end_year_index, run_settings.cadence)

# import all the prior data about the peaks from priors.txt -
# this is estimated fit values and the upper and lower bounds on the values
all_peaks = import_priors(priors_filename, run_settings)

# find which peaks need to be fitted together e.g. l=0/2 and l=1/3 peaks
# returns list made up of the data for each pair of peaks
peak_pairs = find_pairs(all_peaks)

# iterate through all pairs of neighbouring l=0/2 and l=1/3 peaks and fit them
for peak_pair in peak_pairs:
    # fit_info contains all the priors information for a fit of a pair of peaks
    # this includes information about the pair of peaks and the background
    background_prior = copy.deepcopy(run_settings.background) # deep copy stops past fits changing future background priors
    fit_info = Fit_Info(peak_pair_list=[peak_pair], background = background_prior,
                        visibilities=run_settings.visibilities)

    # read the expected frequencies of the peaks (these come from the priors)
    peak_freqs = fit_info.freq_range()

    middle_of_fit_window = sum(peak_freqs)/len(peak_freqs)

    # compares peak frequencies from the priors to see if the peaks are in the frequency range
    # that you want to fit
    if run_settings.lowest_freq_fitted < middle_of_fit_window < run_settings.highest_freq_fitted:
        # SELECT THE FREQUENCY RANGE OVER WHICH THE POWER ARRAY IS FITTED

        # ensures the peaks in the array are in increasing frequency order
        lowest_freq = peak_freqs[0] - run_settings.window_width # lowest bound in frequency of fit window
        highest_freq = peak_freqs[-1] + run_settings.window_width # top bound in frequency of fit window
        lowest_freq_index = np.argmax(freq > lowest_freq)
        highest_freq_index = np.argmax(freq > highest_freq)

        # arrays to be fitted for this pair of peaks
        freq_peaks = freq[lowest_freq_index: highest_freq_index]
        power_peaks = power[lowest_freq_index: highest_freq_index]

        # set the initial parameters for the MCMC walkers
        pos = walker_pos(run_settings.nwalkers, fit_info.lower_prior, fit_info.upper_prior)
        ndim = len(fit_info.fit_pars)
        # intialise MCMC sampler
        sampler = emcee.EnsembleSampler(run_settings.nwalkers, ndim, log_probability,
                                        args=(freq_peaks, power_peaks, fit_info))

        # run burnin to make sure walkers begin from good parameter values
        state = sampler.run_mcmc(pos, run_settings.burnin, progress=True)
        sampler.reset()

        # collect samples used to find the fit values
        state1 = sampler.run_mcmc(state, run_settings.no_samples, progress=True)
        samples = sampler.flatchain

        # save all output of fit - graphs, fit parameter values
        save_output(samples, fit_info.par_labels, directory, fit_info, freq_peaks, power_peaks)
        del fit_info
        del sampler
