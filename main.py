import math
import multiprocessing
import os
import time
import copy

import corner
import emcee
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from dataclasses import dataclass

from import_data import import_data
import model
from probability_functions import log_probability
from structures import Peak_Structure as pk
from structures import Run_Info, Fit_Info, Parameter
from import_priors import import_priors, find_pairs
from initialise_walker_positions import walker_pos
from output import create_directory, save_output

background = Parameter(50000, 0, 100000, False, False)
run_info = Run_Info(background)
directory = create_directory("GOLF")

filename = "GOLF_bw_rw_pm3_960411_200830_Sync_RW_dt60F_v1.fits"
priors_filename = "priors.txt"
data_year = 31536000/run_info.cadence
power, freq = import_data(filename, 0, data_year, run_info.cadence)
all_peaks = import_priors(priors_filename, run_info)
peak_pairs = find_pairs(all_peaks)


for peak_pair in peak_pairs:
    fit_info = Fit_Info([peak_pair], copy.deepcopy(run_info.background))
    fit_info.get_all_fit_variables()
    peak_freqs = fit_info.freq_range()
    if run_info.lower_range < sum(peak_freqs)/len(peak_freqs) < run_info.upper_range:
        peak_freqs.sort()
        lowest_freq = peak_freqs[0] - run_info.window_width
        highest_freq = peak_freqs[-1] + run_info.window_width
        LowerPos = np.argmax(freq > lowest_freq)
        UpperPos = np.argmax(freq > highest_freq)
        freq_peaks = freq[LowerPos: UpperPos]
        power_peaks = power[LowerPos: UpperPos]

        pos = walker_pos(run_info.nwalkers, fit_info.lower_prior, fit_info.upper_prior)
        ndim = len(fit_info.fit_pars)
        sampler = emcee.EnsembleSampler(run_info.nwalkers, ndim, log_probability,
                                        args=(freq_peaks, power_peaks, fit_info))

        state = sampler.run_mcmc(pos, 1000, progress=True)
        sampler.reset()

        state1 = sampler.run_mcmc(state, 10000, progress=True)
        samples = sampler.flatchain
        save_output(samples, fit_info.par_labels, directory, fit_info, freq_peaks, power_peaks)
        del fit_info
        del sampler
